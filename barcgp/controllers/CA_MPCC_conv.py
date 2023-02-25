#!/usr/bin python3

import numpy as np
import scipy as sp

import casadi as ca
import cvxpy as cp
import hpipm_python as hp

import os, sys, time, pathlib, shutil, copy

from typing import Tuple, List, Dict
from collections import deque

from barcgp.dynamics.models.dynamics_models import CasadiDynamicsModel
from barcgp.common.pytypes import VehicleState, VehicleActuation, VehiclePrediction

from barcgp.controllers.abstract_controller import AbstractController
from barcgp.controllers.utils.controllerTypes import CAMPCCParams

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb

class CA_MPCC_conv(AbstractController):
    def __init__(self, dynamics: CasadiDynamicsModel, 
                       costs: Dict[str, List[ca.Function]], 
                       constraints: Dict[str, ca.Function],
                       bounds: Dict[str, VehicleState],
                       control_params: CAMPCCParams=CAMPCCParams(),
                       print_method=print):
        self.dynamics       = dynamics
        self.dt             = dynamics.dt
        self.track          = dynamics.track
        self.costs          = costs
        self.constraints    = constraints

        self.verbose        = control_params.verbose

        if print_method is None:
            self.print_method = lambda s: None
        else:
            self.print_method = print_method

        self.L              = self.track.track_length
        self.n_u            = self.dynamics.n_u + 1
        self.n_q            = self.dynamics.n_q + 1

        self.N              = control_params.N

        self.soft_track     = control_params.soft_track

        self.pos_idx        = [3, 4] # control_params.pos_idx

        # List of indexes for the constraints at each time step
        self.soft_constraint_idxs   = control_params.soft_constraint_idxs
        if self.soft_constraint_idxs is not None:
            self.soft_constraint_quad      = [np.array(q) for q in control_params.soft_constraint_quad]
            self.soft_constraint_lin       = [np.array(l) for l in control_params.soft_constraint_lin]

        if control_params.state_scaling:
            self.state_scaling = 1/np.array(control_params.state_scaling)
        else:
            self.state_scaling = np.ones(self.dynamics.n_q)
        if control_params.input_scaling:
            self.input_scaling  = 1/np.array(control_params.input_scaling)
        else:
            self.input_scaling = np.ones(self.dynamics.n_u)
        self.s_scaling      = 1/np.ceil(self.track.track_length)
        self.vs_scaling     = 1/max(np.abs(control_params.vs_max), np.abs(control_params.vs_min))

        self.damping            = control_params.damping
        self.qp_iters           = control_params.qp_iters
        self.qp_interface       = control_params.qp_interface

        self.track_tightening   = control_params.track_tightening

        self.parametric_contouring_cost   = control_params.parametric_contouring_cost
        if self.parametric_contouring_cost:
            self.q_c    = ca.MX.sym('q_c', 1)
            self.q_cN   = ca.MX.sym('q_cN', 1)
        else:
            self.q_c    = control_params.contouring_cost
            self.q_cN   = control_params.contouring_cost_N

        self.q_l            = control_params.lag_cost
        self.q_lN           = control_params.lag_cost_N
        self.q_p            = control_params.performance_cost
        self.q_v            = control_params.vs_cost
        self.q_dv           = control_params.vs_rate_cost

        self.q_tq           = control_params.track_slack_quad
        self.q_tl           = control_params.track_slack_lin

        self.vs_max         = control_params.vs_max
        self.vs_min         = control_params.vs_min
        self.vs_rate_max    = control_params.vs_rate_max
        self.vs_rate_min    = control_params.vs_rate_min

        self.delay          = control_params.delay
        self.delay_buffer   = []
        if self.delay is None:
            self.delay = np.zeros(self.dynamics.n_u)
            self.delay_buffer = None

        self.solver_name    = control_params.solver_name
        self.code_gen       = control_params.code_gen
        self.jit            = control_params.jit
        self.opt_flag       = control_params.opt_flag

        if self.code_gen:
            self.c_file_name = self.solver_name + '.c'
            self.so_file_name = self.solver_name + '.so'
            if control_params.solver_dir is not None:
                self.solver_dir = pathlib.Path(control_params.solver_dir).expanduser().joinpath(self.solver_name)
        
        # Process box constraints
        self.state_ub, self.input_ub = self.dynamics.state2qu(bounds['qu_ub'])
        self.state_lb, self.input_lb = self.dynamics.state2qu(bounds['qu_lb'])
        _, self.input_rate_ub = self.dynamics.state2qu(bounds['du_ub'])
        _, self.input_rate_lb = self.dynamics.state2qu(bounds['du_lb'])

        self.x_ub  = np.append(self.state_ub, 2*self.L)
        self.x_lb  = np.append(self.state_lb, -1)
        self.w_ub  = np.append(self.input_ub, self.vs_max)
        self.w_lb  = np.append(self.input_lb, self.vs_min)
        self.dw_ub = np.append(self.input_rate_ub, self.vs_rate_max)
        self.dw_lb = np.append(self.input_rate_lb, self.vs_rate_min)

        self.ub = np.concatenate((np.tile(np.concatenate((self.x_ub, self.w_ub)), self.N+1), np.tile(self.dw_ub, self.N)))
        self.lb = np.concatenate((np.tile(np.concatenate((self.x_lb, self.w_lb)), self.N+1), np.tile(self.dw_lb, self.N)))
        # if self.soft_track and self.qp_interface in ['casadi']:
        #     self.ub = np.concatenate((self.ub, np.inf*np.ones(2)))
        #     self.lb = np.concatenate((self.lb, np.zeros(2)))
        
        self.n_c = [0 for _ in range(self.N+1)]
        self.n_sc = [0 for _ in range(self.N+1)]
        if self.soft_constraint_idxs is not None:
            self.n_sc = [len(si) for si in self.soft_constraint_idxs]

        # Construct normalization matrix
        self.x_scaling      = np.append(self.state_scaling, self.s_scaling)
        self.x_scaling_inv  = 1/self.x_scaling
        self.w_scaling      = np.append(self.input_scaling, self.vs_scaling)
        self.w_scaling_inv  = 1/self.w_scaling
        
        self.xw_scaling     = np.concatenate((self.x_scaling, self.w_scaling))
        self.xw_scaling_inv = 1/self.xw_scaling

        self.D_scaling = np.concatenate((np.tile(np.concatenate((self.x_scaling, self.w_scaling)), self.N+1), np.ones(self.N*self.n_u)))
        self.D_scaling_inv = 1/self.D_scaling

        self.q_pred = np.zeros((self.N+1, self.dynamics.n_q))
        self.s_pred = np.zeros(self.N+1)

        self.u_pred = np.zeros((self.N, self.dynamics.n_u))
        self.vs_pred = np.zeros(self.N)

        self.du_pred = np.zeros((self.N, self.dynamics.n_u))
        self.dvs_pred = np.zeros(self.N)

        self.u_prev = np.zeros(self.dynamics.n_u)
        self.vs_prev = 0

        self.dw_pred = np.zeros((self.N, self.n_u))

        self.u_ws = None
        self.vs_ws = None

        self.du_ws = None
        self.dvs_ws = None

        self.l_ws = None

        self.state_input_prediction = VehiclePrediction()

        self.initialized = False

        self._build_solver()

        # self.l_pred = np.zeros(self.n_c)

        self.debug = False
        self.debug_plot = control_params.debug_plot
        if self.debug_plot:
            plt.ion()
            self.fig = plt.figure(figsize=(10,5))
            self.ax_xy = self.fig.add_subplot(1,2,1)
            self.ax_a = self.fig.add_subplot(3,2,2)
            self.ax_d = self.fig.add_subplot(3,2,4)
            self.ax_v = self.fig.add_subplot(3,2,6)
            # self.joint_dynamics.dynamics_models[0].track.remove_phase_out()
            self.dynamics.track.plot_map(self.ax_xy, close_loop=False)
            # self.colors = ['b', 'g', 'r', 'm', 'c']
            self.l_xy = self.ax_xy.plot([], [], 'bo', markersize=4)[0]
            self.l_c = self.ax_xy.plot([], [], 'ko', markersize=2)[0]
            self.l_i = self.ax_xy.plot([], [], 'go', markersize=2)[0]
            self.l_o = self.ax_xy.plot([], [], 'go', markersize=2)[0]
            self.l_a = self.ax_a.plot([], [], '-bo')[0]
            self.l_d = self.ax_d.plot([], [], '-bo')[0]
            self.l_v = self.ax_v.plot([], [], '-bo')[0]
            self.ax_a.set_ylabel('accel')
            self.ax_d.set_ylabel('steering')
            self.ax_v.set_ylabel('v')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def _update_debug_plot(self, q, s, u, v):
        self.l_xy.set_data(q[:,3], q[:,4])
        self.l_c.set_data(self.x_s(np.mod(s, self.L)).full(), self.y_s(np.mod(s, self.L)).full())
        self.l_i.set_data(self.xi_s(np.mod(s, self.L)).full(), self.yi_s(np.mod(s, self.L)).full())
        self.l_o.set_data(self.xo_s(np.mod(s, self.L)).full(), self.yo_s(np.mod(s, self.L)).full())
        self.ax_xy.set_aspect('equal')
        self.l_a.set_data(np.arange(self.N), u[:,0])
        self.l_d.set_data(np.arange(self.N), u[:,1])
        self.l_v.set_data(np.arange(self.N), v)
        self.ax_a.relim()
        self.ax_a.autoscale_view()
        self.ax_d.relim()
        self.ax_d.autoscale_view()
        self.ax_v.relim()
        self.ax_v.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def initialize(self):
        pass

    def step(self, vehicle_state: VehicleState, 
                reference: np.ndarray = np.array([]),
                parameters: np.ndarray = np.array([])):
        self.solve(vehicle_state, reference, parameters)

        # u = np.array([self.u_pred[int(self.delay[i]),i] for i in range(self.dynamics.n_u)])
        u = self.u_pred[0]
        self.dynamics.qu2state(vehicle_state, None, u)
        self.dynamics.qu2prediction(self.state_input_prediction, self.q_pred, self.u_pred)
        self.state_input_prediction.t = vehicle_state.t

        # Update delay buffer
        if self.delay_buffer is not None:
            for i in range(self.dynamics.n_u):
                self.delay_buffer[i].append(u[i])

        # Construct initial guess for next iteration
        u_ws = np.vstack((self.u_pred, self.u_pred[-1]))
        vs_ws = np.append(self.vs_pred, self.vs_pred[-1])
        du_ws = np.vstack((self.du_pred[1:], self.du_pred[-1]))
        dvs_ws = np.append(self.dvs_pred[1:], self.dvs_pred[-1])
        self.set_warm_start(u_ws, vs_ws, du_ws, dvs_ws)

        return

    def set_warm_start(self, u_ws: np.ndarray, vs_ws: np.ndarray, du_ws: np.ndarray, dvs_ws: np.ndarray, 
                       state: VehicleState = None, 
                       l_ws: np.ndarray = None,
                       reference: np.ndarray = np.array([]),
                       parameters: np.ndarray = np.array([])):
        self.u_ws = u_ws
        self.vs_ws = vs_ws

        self.u_prev = u_ws[0]
        self.vs_prev = vs_ws[0]

        self.du_ws = du_ws
        self.dvs_ws = dvs_ws

        if l_ws is not None:
            self.l_ws = l_ws

        if not self.initialized and state:
            state.e.psi = np.mod(state.e.psi, 2*np.pi)
            # Default values for reference interpolation
            if len(reference) == 0:
                reference = np.zeros(self.N+1)

            q0, _ = self.dynamics.state2qu(state)
            s0, _, _ = self.track.global_to_local((state.x.x, state.x.y, 0))
            q_ws, s_ws = self._evaluate_dynamics(q0, s0, self.u_ws[1:], self.vs_ws[1:])

            D = np.concatenate((np.hstack((q_ws, s_ws.reshape((-1,1)), self.u_ws, self.vs_ws.reshape((-1,1)))).ravel(), 
                                np.hstack((self.du_ws, self.dvs_ws.reshape((-1,1)))).ravel()))
            P = np.concatenate((q0, [s0], self.u_prev, [self.vs_prev], s_ws, reference, parameters))
            # damping = np.concatenate((0.5*np.ones((self.N+1)*(self.n_q+self.n_u)), np.zeros(self.N*self.n_u)))
            damping = 0.5
            
            # if self.soft_track and self.qp_interface in ['casadi']:
            #     D = np.concatenate((D, np.zeros(2)))
                # damping = np.concatenate((damping, np.zeros(2)))

            for _ in range(5):
                # Evaluate QP approximation
                if self.qp_interface == 'casadi':
                    D_bar, success, status = self._solve_casadi(D, P)
                elif self.qp_interface == 'hpipm':
                    D_bar, success, status = self._solve_hpipm(D, P)
                elif self.qp_interface == 'cvxpy':
                    D_bar, success, status = self._solve_cvxpy(D, P)
                if not success:
                    self.print_method('QP returned ' + str(status))
                    break
                D = damping*D + (1-damping)*D_bar
                D[(self.N+1)*(self.n_q+self.n_u):] = 0

                if self.debug:
                    xw_bar = D_bar[:(self.n_q+self.n_u)*(self.N+1)].reshape((self.N+1, self.n_q+self.n_u))
                    dw_bar = D_bar[(self.n_q+self.n_u)*(self.N+1):(self.n_q+self.n_u)*(self.N+1)+self.n_u*self.N].reshape((self.N, self.n_u))
                    xw = D[:(self.n_q+self.n_u)*(self.N+1)].reshape((self.N+1, self.n_q+self.n_u))
                    dw = D[(self.n_q+self.n_u)*(self.N+1):(self.n_q+self.n_u)*(self.N+1)+self.n_u*self.N].reshape((self.N, self.n_u))
                    # pdb.set_trace()

            xw_sol = D[:(self.n_q+self.n_u)*(self.N+1)].reshape((self.N+1, self.n_q+self.n_u))
            dw_sol = D[(self.n_q+self.n_u)*(self.N+1):(self.n_q+self.n_u)*(self.N+1)+self.n_u*self.N].reshape((self.N, self.n_u))
            
            if self.debug_plot:
                self._update_debug_plot(xw_sol[:,:self.n_q-1], xw_sol[:,self.n_q-1], xw_sol[1:,self.n_q:self.n_q+self.n_u-1], xw_sol[1:,self.n_q+self.n_u-1])

            self.u_ws, self.vs_ws = xw_sol[:,self.n_q:self.n_q+self.dynamics.n_u], xw_sol[:,-1]
            self.du_ws, self.dvs_ws = dw_sol[:,:self.dynamics.n_u], dw_sol[:,self.dynamics.n_u:]

            q_sol, s_sol = self._evaluate_dynamics(q0, s0, self.u_ws[1:], self.vs_ws[1:])

            self.q_pred = q_sol
            self.s_pred = s_sol
            self.u_pred = self.u_ws[1:]
            self.vs_pred = self.vs_ws[1:]
            self.du_pred = self.du_ws
            self.dvs_pred = self.dvs_ws

            if self.delay_buffer is not None:
                for i in range(self.dynamics.n_u):
                    self.delay_buffer.append(deque(self.u_ws[1:1+self.delay[i],i], maxlen=self.delay[i]))
            self.initalized = True

    def _evaluate_dynamics(self, q0, s0, U, VS):
        t = time.time()
        Q, S = [q0], [s0]
        for k in range(U.shape[0]):
            Q.append(self.dynamics.fd(Q[k], U[k]).toarray().squeeze())
            S.append(S[k] + self.dt*VS[k])
        if self.verbose:
            self.print_method('Dynamics evalution time: ' + str(time.time()-t))
        return np.array(Q), np.array(S)

    def solve(self, state: VehicleState,
              reference: np.ndarray = np.array([]),
              parameters: np.ndarray = np.array([])):
        state.e.psi = np.mod(state.e.psi, 2*np.pi)
        # Default values for reference interpolation
        if len(reference) == 0:
            reference = np.zeros(self.N+1)
        
        q0, _ = self.dynamics.state2qu(state)
        s0, _, _ = self.track.global_to_local((state.x.x, state.x.y, 0))

        if self.delay_buffer is not None:
            delay_steps = int(np.amin(self.delay))
            u_delay = np.hstack([np.array(self.delay_buffer[i])[:delay_steps].reshape((-1,1)) for i in range(self.dynamics.n_u)])
            q_bar, s_bar = self._evaluate_dynamics(q0, s0, u_delay, self.vs_ws[1:1+delay_steps])
            q0, s0 = q_bar[-1], s_bar[-1]

        q_ws, s_ws = self._evaluate_dynamics(q0, s0, self.u_ws[1:], self.vs_ws[1:])

        D = np.concatenate((np.hstack((q_ws, s_ws.reshape((-1,1)), self.u_ws, self.vs_ws.reshape((-1,1)))).ravel(), 
                            np.hstack((self.du_ws, self.dvs_ws.reshape((-1,1)))).ravel()))
        P = np.concatenate((q0, [s0], self.u_prev, [self.vs_prev], s_ws, reference, parameters))

        # if self.soft_track and self.qp_interface in ['casadi']:
        #     D = np.concatenate((D, np.zeros(2)))
        
        for _ in range(self.qp_iters):
            if self.qp_interface == 'casadi':
                D_bar, success, status = self._solve_casadi(D, P)
            elif self.qp_interface == 'hpipm':
                D_bar, success, status = self._solve_hpipm(D, P)
            elif self.qp_interface == 'cvxpy':
                D_bar, success, status = self._solve_cvxpy(D, P)
            if not success:
                self.print_method('QP returned ' + str(status))
                break

            D = self.damping*D + (1-self.damping)*D_bar
            D[(self.N+1)*(self.n_q+self.n_u):] = 0

        if success:
            # Unpack solution
            xw_sol = D[:(self.n_q+self.n_u)*(self.N+1)].reshape((self.N+1, self.n_q+self.n_u))
            dw_sol = D[(self.n_q+self.n_u)*(self.N+1):(self.n_q+self.n_u)*(self.N+1)+self.n_u*self.N].reshape((self.N, self.n_u))

            u_sol, vs_sol = xw_sol[1:,self.n_q:self.n_q+self.dynamics.n_u], xw_sol[1:,-1]
            du_sol, dvs_sol = dw_sol[:,:self.dynamics.n_u], dw_sol[:,self.dynamics.n_u:]

        if not success:
            u_sol, vs_sol = self.u_ws[1:], self.vs_ws[1:]
            du_sol, dvs_sol = self.du_ws, self.dvs_ws

        q_sol, s_sol = self._evaluate_dynamics(q0, s0, u_sol, vs_sol)
        if self.debug_plot:
            self._update_debug_plot(q_sol, s_sol, u_sol, vs_sol)
        # self.l_pred = np.zeros(self.n_c+self.n_q*self.N)    

        self.q_pred = q_sol
        self.s_pred = s_sol
        self.u_pred = u_sol
        self.vs_pred = vs_sol
        self.du_pred = du_sol
        self.dvs_pred = dvs_sol

    def get_prediction(self):
        return self.state_input_prediction

    @staticmethod
    def get_casadi_error_function(track):
        L = track.track_length
        S = np.linspace(0, L, 100)
        X, Y = [], []
        for s in S:
            # Centerline
            x, y, _ = track.local_to_global((s, 0, 0))
            X.append(x)
            Y.append(y)
        x_s = ca.interpolant('x_s', 'bspline', [S], X)
        y_s = ca.interpolant('y_s', 'bspline', [S], Y)

        s_sym = ca.MX.sym('s', 1)
        dsdx_s = ca.Function('dsdx_s', [s_sym], [ca.jacobian(x_s(s_sym), s_sym)])
        dsdy_s = ca.Function('dsdy_s', [s_sym], [ca.jacobian(y_s(s_sym), s_sym)])

        p_sym = ca.MX.sym('p', 2)

        s_mod = ca.mod(s_sym, L)
        t = ca.atan2(dsdy_s(s_mod), dsdx_s(s_mod))
        ec =  ca.sin(t)*(p_sym[0]-x_s(s_mod)) - ca.cos(t)*(p_sym[1]-y_s(s_mod))
        el = -ca.cos(t)*(p_sym[0]-x_s(s_mod)) - ca.sin(t)*(p_sym[1]-y_s(s_mod))
        f_e = ca.Function('e', [p_sym, s_sym], [ca.vertcat(ec, el)])

        return f_e

    def _build_solver(self):
        # Compute spline approximation of track
        # S = np.linspace(-self.track.track_length, 2*self.track.track_length, 100)
        S = np.linspace(0, self.track.track_length, 100)
        X, Y, Xi, Yi, Xo, Yo = [], [], [], [], [], []
        for s in S:
            # Centerline
            x, y, _ = self.track.local_to_global((s, 0, 0))
            X.append(x)
            Y.append(y)
            # Inside boundary
            xi, yi, _ = self.track.local_to_global((s, self.track.half_width-self.track_tightening, 0))
            Xi.append(xi)
            Yi.append(yi)
            # Outside boundary
            xo, yo, _ = self.track.local_to_global((s, -(self.track.half_width-self.track_tightening), 0))
            Xo.append(xo)
            Yo.append(yo)
        self.x_s = ca.interpolant('x_s', 'bspline', [S], X)
        self.y_s = ca.interpolant('y_s', 'bspline', [S], Y)
        self.xi_s = ca.interpolant('xi_s', 'bspline', [S], Xi)
        self.yi_s = ca.interpolant('yi_s', 'bspline', [S], Yi)
        self.xo_s = ca.interpolant('xo_s', 'bspline', [S], Xo)
        self.yo_s = ca.interpolant('yo_s', 'bspline', [S], Yo)

        # Compute derivatives of track
        s_sym = ca.MX.sym('s', 1)
        self.dsdx_s = ca.Function('dsdx_s', [s_sym], [ca.jacobian(self.x_s(s_sym), s_sym)])
        self.dsdy_s = ca.Function('dsdy_s', [s_sym], [ca.jacobian(self.y_s(s_sym), s_sym)])
        
        # Dynamcis augmented with arc length dynamics
        q_sym = ca.MX.sym('q', self.dynamics.n_q)
        u_sym = ca.MX.sym('u', self.dynamics.n_u)
        vs_sym = ca.MX.sym('vs', 1)

        x_sym = ca.vertcat(q_sym, s_sym)
        w_sym = ca.vertcat(u_sym, vs_sym)

        n_z = self.n_q + self.n_u

        # Linear approximation of the continuous time dynamics
        # Ac = ca.jacobian(self.dynamics.fc(q_sym, u_sym), q_sym)
        # Bc = ca.jacobian(self.dynamics.fc(q_sym, u_sym), u_sym)
        # gc = self.dynamics.fc(q_sym, u_sym) - Ac @ q_sym - Bc @ u_sym

        # # Exact disretization with zero-order hold
        # H = self.dt*ca.vertcat(ca.horzcat(Ac, Bc, gc), ca.DM.zeros(self.dynamics.n_u+1, self.dynamics.n_q+self.dynamics.n_u+1))
        # M = ca.expm(H)
        # Ad = ca.MX.sym('Ad', ca.Sparsity(self.n_q, self.n_q))
        # Ad[:self.dynamics.n_q,:self.dynamics.n_q] = M[:self.dynamics.n_q,:self.dynamics.n_q]
        # Ad[-1,-1] = 1
        # Bd = ca.MX.sym('Bd', ca.Sparsity(self.n_q, self.n_u))
        # Bd[:self.dynamics.n_q,:self.dynamics.n_u] = M[:self.dynamics.n_q,self.dynamics.n_q:self.dynamics.n_q+self.dynamics.n_u]
        # Bd[-1,-1] = self.dt
        # gd = ca.vertcat(M[:self.dynamics.n_q,self.dynamics.n_q+self.dynamics.n_u], ca.DM.zeros(1))
        # f_Ad = ca.Function('Ad', [x_sym, w_sym], [Ad])
        # f_Bd = ca.Function('Bd', [x_sym, w_sym], [Bd])
        # f_gd = ca.Function('gd', [x_sym, w_sym], [gd])

        # Use default discretization scheme
        Ad = ca.MX.sym('Ad', ca.Sparsity(self.n_q, self.n_q))
        Ad[:self.dynamics.n_q,:self.dynamics.n_q] = self.dynamics.fAd(q_sym, u_sym)
        Ad[-1,-1] = 1
        Bd = ca.MX.sym('Bd', ca.Sparsity(self.n_q, self.n_u))
        Bd[:self.dynamics.n_q,:self.dynamics.n_u] = self.dynamics.fBd(q_sym, u_sym)
        Bd[-1,-1] = self.dt
        gd = ca.vertcat(self.dynamics.fd(q_sym, u_sym) - self.dynamics.fAd(q_sym, u_sym) @ q_sym - self.dynamics.fBd(q_sym, u_sym) @ u_sym, ca.DM.zeros(1))
        f_Ad = ca.Function('Ad', [x_sym, w_sym], [Ad])
        f_Bd = ca.Function('Bd', [x_sym, w_sym], [Bd])
        f_gd = ca.Function('gd', [x_sym, w_sym], [gd])

        # Contouring and lag errors and their gradients
        s_mod = ca.mod(s_sym, self.L)
        # Interpolation variable in range [-1, 1] (outside, inside)
        z_sym = ca.MX.sym('z', 1)
        t = ca.atan2(self.dsdy_s(s_mod), self.dsdx_s(s_mod))
        # ec =  ca.sin(t)*(x_sym[self.pos_idx[0]]-self.x_s(s_mod)) - ca.cos(t)*(x_sym[self.pos_idx[1]]-self.y_s(s_mod))
        # el = -ca.cos(t)*(x_sym[self.pos_idx[0]]-self.x_s(s_mod)) - ca.sin(t)*(x_sym[self.pos_idx[1]]-self.y_s(s_mod))
        # f_e = ca.Function('ec', [x_sym], [ca.vertcat(ec, el)])
        # f_dx_e = ca.Function('ds_ec', [x_sym], [ca.jacobian(ca.vertcat(ec, el), x_sym)])
        x_int = self.xo_s(s_mod) + (z_sym+1)/2*(self.xi_s(s_mod)-self.xo_s(s_mod))
        y_int = self.yo_s(s_mod) + (z_sym+1)/2*(self.yi_s(s_mod)-self.yo_s(s_mod))
        ec =  ca.sin(t)*(x_sym[self.pos_idx[0]]-x_int) - ca.cos(t)*(x_sym[self.pos_idx[1]]-y_int)
        el = -ca.cos(t)*(x_sym[self.pos_idx[0]]-x_int) - ca.sin(t)*(x_sym[self.pos_idx[1]]-y_int)
        f_e = ca.Function('ec', [x_sym, z_sym], [ca.vertcat(ec, el)])
        f_dx_e = ca.Function('ds_ec', [x_sym, z_sym], [ca.jacobian(ca.vertcat(ec, el), x_sym)])
        
        # q_0, ..., q_N
        q_ph = [ca.MX.sym(f'q_ph_{k}', self.dynamics.n_q) for k in range(self.N+1)] # State
        s_ph = [ca.MX.sym(f's_{k}', 1) for k in range(self.N+1)]
        x_ph = [ca.vertcat(q_ph[k], s_ph[k]) for k in range(self.N+1)]

        # u_-1, u_0, ..., u_N-1
        u_ph = [ca.MX.sym(f'u_ph_{k}', self.dynamics.n_u) for k in range(self.N+1)] # Inputs
        vs_ph = [ca.MX.sym(f'vs_{k}', 1) for k in range(self.N+1)]
        w_ph = [ca.vertcat(u_ph[k], vs_ph[k]) for k in range(self.N+1)]

        # du_0, ..., du_N-1
        du_ph = [ca.MX.sym(f'du_ph_{k}', self.dynamics.n_u) for k in range(self.N)] # Input rates
        dvs_ph = [ca.MX.sym(f'dvs_{k}', 1) for k in range(self.N)]
        dw_ph = [ca.vertcat(du_ph[k], dvs_ph[k]) for k in range(self.N)]
        
        if self.soft_track:
            si_ph = ca.MX.sym('in_bound_slack_ph')
            so_ph = ca.MX.sym('out_bound_slack_ph')

        # Parameters
        sp_ph = [ca.MX.sym(f'sp_{k}', 1) for k in range(self.N+1)] # Approximate progress from previous time step
        xw0_ph = ca.MX.sym('xw0', n_z) # Initial state
        z_ph = ca.MX.sym('z_ph', self.N+1)

        # Scaling matricies
        T_x = ca.DM(sp.sparse.diags(self.x_scaling))
        T_x_inv = ca.DM(sp.sparse.diags(self.x_scaling_inv))
        T_w = ca.DM(sp.sparse.diags(self.w_scaling))
        T_w_inv = ca.DM(sp.sparse.diags(self.w_scaling_inv))
        T_xw = ca.DM(sp.sparse.diags(np.concatenate((self.x_scaling, self.w_scaling))))
        T_xw_inv = ca.DM(sp.sparse.diags(np.concatenate((self.x_scaling_inv, self.w_scaling_inv))))

        state_cost_params = []
        input_cost_params = []
        rate_cost_params = []
        constraint_params = []

        A, B, g = [], [], []
        Q_xw, q_xw, Q_dw, q_dw = [], [], [], []
        C_xw, C_xw_lb, C_xw_ub, C_dw, C_dw_lb, C_dw_ub = [], [], [], [], [], []
        if self.soft_track and self.qp_interface in ['casadi']:
            C_xw_s = []
        for k in range(self.N+1):
            _Q_xw = ca.MX.sym(f'_Q_xw{k}', ca.Sparsity(n_z, n_z))
            _q_xw = ca.MX.sym(f'_q_xw_{k}', ca.Sparsity(n_z, 1))

            if k < self.N:
                P_cl = ca.diag(ca.vertcat(self.q_c, self.q_l))
            else:
                P_cl = ca.diag(ca.vertcat(self.q_cN, self.q_lN))

            # Quadratic approximation of lag and contouring costs
            e = f_e(x_ph[k], z_ph[k])
            Dx_e = f_dx_e(x_ph[k], z_ph[k])
            Q_e = Dx_e.T @ P_cl @ Dx_e
            q_e = 2* Dx_e.T @ P_cl @ e - 2 * Q_e @ x_ph[k]
            
            # Quadratic approximation of state costs
            if self.costs['state'][k]:
                if self.costs['state'][k].n_in() == 2:
                    pq_k = ca.MX.sym(f'pq_{k}', self.costs['state'][k].numel_in(1))
                    Jx_k = self.costs['state'][k](q_ph[k], pq_k)
                    state_cost_params.append(pq_k)
                else:
                    Jx_k = self.costs['state'][k](q_ph[k])
            else:
                Jx_k = ca.DM.zeros(1)
            M_x = ca.jacobian(ca.jacobian(Jx_k, x_ph[k]), x_ph[k])
            m_x = ca.jacobian(Jx_k, x_ph[k]).T
            _Q_xw[:self.n_q,:self.n_q] = 2 * T_x_inv @ (Q_e + M_x) @ T_x_inv
            _q_xw[:self.n_q] = T_x_inv @ (q_e + m_x - M_x @ x_ph[k])

            # Quadratic approximation of input costs
            if self.costs['input'][k]:
                if self.costs['input'][k].n_in() == 2:
                    pu_k = ca.MX.sym(f'pu_{k}', self.costs['input'][k].numel_in(1))
                    Jw_k = self.costs['input'][k](u_ph[k])
                    input_cost_params.append(pu_k)
                else:    
                    Jw_k = self.costs['input'][k](u_ph[k])
            else:
                Jw_k = ca.DM.zeros(1)
            Jw_k += 0.5*self.q_v*vs_ph[k]**2 - self.q_p*vs_ph[k]
            M_w = ca.jacobian(ca.jacobian(Jw_k, w_ph[k]), w_ph[k])
            m_w = ca.jacobian(Jw_k, w_ph[k]).T
            _Q_xw[self.n_q:,self.n_q:] = 2 * T_w_inv @ M_w @ T_w_inv
            _q_xw[self.n_q:] = T_w_inv @ (m_w - M_w @ w_ph[k])

            Q_xw.append((_Q_xw + _Q_xw.T)/2 + 1e-10*ca.DM.eye(self.n_q+self.n_u))
            q_xw.append(_q_xw)

            _C_xw, _C_xw_ub, _C_xw_lb = ca.MX.sym(f'_C_xw_{k}', 0, n_z), ca.MX.sym(f'_C_xw_ub_{k}', 0), ca.MX.sym(f'_C_xw_lb_{k}', 0)
            if self.soft_track and self.qp_interface in ['casadi']:
                _C_xw_s = ca.MX.sym(f'_C_xw_s_{k}', 0, 2)
            if k >= 1:
                # Linear approximation of track boundary constraints
                xi, yi = self.xi_s(ca.mod(sp_ph[k], self.L)), self.yi_s(ca.mod(sp_ph[k], self.L))
                xo, yo = self.xo_s(ca.mod(sp_ph[k], self.L)), self.yo_s(ca.mod(sp_ph[k], self.L))
                n, d = -(xo - xi), yo - yi

                if self.soft_track and self.qp_interface in ['casadi']:
                    _C_xw = ca.MX.sym(f'_C_xw_{k}', ca.Sparsity(2, n_z))
                    _C_xw_ub = ca.MX.sym(f'_C_xw_ub_{k}', ca.Sparsity(2, 1))
                    _C_xw_lb = ca.MX.sym(f'_C_xw_lb_{k}', ca.Sparsity(2, 1))
                    _C_xw_s = ca.MX.sym(f'_C_xw_s_{k}', ca.Sparsity(2, 2))
                    _C_xw[0,self.pos_idx[0]], _C_xw[0,self.pos_idx[1]] = n, -d
                    _C_xw_s[0,0] = -1
                    _C_xw_ub[0] = ca.fmax(n*xi-d*yi, n*xo-d*yo)
                    _C_xw_lb[0] = -np.inf
                    _C_xw[1,self.pos_idx[0]], _C_xw[1,self.pos_idx[1]] = n, -d
                    _C_xw_s[1,1] = 1
                    _C_xw_ub[1] = np.inf
                    _C_xw_lb[1] = ca.fmin(n*xi-d*yi, n*xo-d*yo)
                else:
                    _C_xw = ca.MX.sym(f'_C_xw_{k}', ca.Sparsity(1, n_z))
                    # _C_xw = ca.jacobian(n*q_ph[k][self.pos_idx[0]] - d*q_ph[k][self.pos_idx[1]], ca.vertcat(x_ph[k], w_ph[k]))
                    _C_xw[0,self.pos_idx[0]], _C_xw[0,self.pos_idx[1]] = n, -d
                    _C_xw_ub = ca.fmax(n*xi-d*yi, n*xo-d*yo)
                    _C_xw_lb = ca.fmin(n*xi-d*yi, n*xo-d*yo)

            # Linear approximation of constraints on states and inputs
            z_k = ca.vertcat(x_ph[k], w_ph[k])
            if self.constraints['state_input'][k]:
                if self.constraints['state_input'][k].n_in() == 3:
                    pqu_k = ca.MX.sym(f'pqu_{k}', self.constraints['state_input'][k].numel_in(2))
                    C = self.constraints['state_input'][k](q_ph[k], u_ph[k], pqu_k)
                    constraint_params.append(pqu_k)
                else:
                    C = self.constraints['state_input'][k](q_ph[k], u_ph[k])
                _C = ca.jacobian(C, z_k)
                _C_ub = -C + _C @ z_k
                _C_lb = -1e10*np.ones(_C_ub.size1()) #-ca.DM.inf(_C_ub.size1())
                
                _C_xw = ca.vertcat(_C_xw, _C)
                _C_xw_ub = ca.vertcat(_C_xw_ub, _C_ub)
                _C_xw_lb = ca.vertcat(_C_xw_lb, _C_lb)
            if self.soft_track and self.qp_interface in ['casadi']:
                _C_xw_s = ca.vertcat(_C_xw_s, ca.DM.zeros(_C_xw.size1()-_C_xw_s.size1(), 2))
            
            self.n_c[k] += _C_xw.size1()
            C_xw.append(_C_xw @ T_xw_inv)
            C_xw_ub.append(_C_xw_ub)
            C_xw_lb.append(_C_xw_lb)
            if self.soft_track and self.qp_interface in ['casadi']:
                C_xw_s.append(_C_xw_s)

            if k < self.N:
                # Linearized dynamics
                _A = ca.MX.sym(f'_A_{k}', ca.Sparsity(n_z, n_z))
                _A[:self.n_q,:self.n_q] = T_x @ f_Ad(x_ph[k], w_ph[k]) @ T_x_inv
                _A[:self.n_q,self.n_q:] = T_x @ f_Bd(x_ph[k], w_ph[k]) @ T_w_inv
                _A[self.n_q:,self.n_q:] = ca.DM.eye(self.n_u)
                A.append(_A)

                _B = ca.MX.sym(f'_B_{k}', ca.Sparsity(n_z, self.n_u))
                # _B[:self.n_q,:] = T_x @ f_Bd(x_ph[k], w_ph[k]) @ T_w_inv
                # _B[self.n_q:,:] = ca.DM.eye(self.n_u)
                _B[:self.n_q,:] = self.dt*T_x @ f_Bd(x_ph[k], w_ph[k]) @ T_w_inv
                _B[self.n_q:,:] = self.dt*ca.DM.eye(self.n_u)
                B.append(_B)

                _g = ca.MX.sym(f'_g_{k}', ca.Sparsity(n_z, 1))
                _g[:self.n_q] = T_x @ f_gd(x_ph[k], w_ph[k])
                # _g[self.n_q:] = ca.DM.zeros(self.n_u)
                g.append(_g)

                # Quadratic approximation of input rate costs
                if self.costs['rate'][k]:
                    if self.costs['rate'][k].n_in() == 2:
                        pdu_k = ca.MX.sym(f'pdu_{k}', self.costs['rate'][k].numel_in(1))
                        Jdw_k = self.costs['rate'][k](du_ph[k], pdu_k)
                        rate_cost_params.append(pdu_k)
                    else:
                        Jdw_k = self.costs['rate'][k](du_ph[k])
                else:
                    Jdw_k = ca.DM.zeros(1)
                Jdw_k += 0.5*self.q_dv*dvs_ph[k]**2
                M_dw = ca.jacobian(ca.jacobian(Jdw_k, dw_ph[k]), dw_ph[k])
                m_dw = ca.jacobian(Jdw_k, dw_ph[k]).T
                Q_dw.append(2*M_dw)
                q_dw.append(m_dw - M_dw @ dw_ph[k])

                # Linear approximation of constraints on input rates
                _C_dw, _C_dw_ub, _C_dw_lb = ca.MX.sym(f'_C_xw_{k}', 0, self.n_u), ca.MX.sym(f'_C_xw_ub_{k}', 0), ca.MX.sym(f'_C_xw_lb_{k}', 0)
                if self.constraints['rate'][k]:
                    _C_dw = ca.jacobian(self.constraints['rate'][k](du_ph[k]), dw_ph[k])
                    _C_dw_ub = -self.constraints['rate'][k](du_ph[k]) + _C_dw @ dw_ph[k]
                    _C_dw_lb = -1e10*np.ones(_C_dw.size1()) #-ca.DM.inf(_C_dw_ub.size1())
                
                self.n_c[k] += _C_dw.size1()
                C_dw.append(_C_dw)
                C_dw_ub.append(_C_dw_ub)
                C_dw_lb.append(_C_dw_lb)

        # Form decision vector using augmented states (x_k, w_k) and inputs dw_k
        # D = [(x_0, w_-1), ..., (x_N, w_N-1), dw_0, ..., dw_N-1]
        D = []
        for x, w in zip(x_ph, w_ph):
            D.extend([x, w])
        D += dw_ph
        if self.soft_track and self.qp_interface in ['casadi']:
            D += [si_ph, so_ph]
        D = ca.vertcat(*D)
        n_D = D.size1()

        # Parameters
        P = [ca.vertcat(xw0_ph, *sp_ph, z_ph)] + state_cost_params + input_cost_params + rate_cost_params + constraint_params
        if self.parametric_contouring_cost:
            P += [ca.vertcat(self.q_c, self.q_cN)]
        P = ca.vertcat(*P)

        # Construct QP cost matrix and vector
        H = ca.MX.sym('H', ca.Sparsity(n_D, n_D))
        h = ca.MX.sym('h', ca.Sparsity(n_D, 1))
        for k in range(self.N+1):
            H[k*n_z:(k+1)*n_z,k*n_z:(k+1)*n_z]  = Q_xw[k]
            h[k*n_z:(k+1)*n_z]                  = q_xw[k]
            if k < self.N:
                s_idx, e_idx = (self.N+1)*n_z+k*self.n_u, (self.N+1)*n_z+(k+1)*self.n_u
                H[s_idx:e_idx,s_idx:e_idx]  = Q_dw[k]
                h[s_idx:e_idx]              = q_dw[k]
        if self.soft_track and self.qp_interface in ['casadi']:
            H[-2:,-2:] = self.q_tq*ca.DM.eye(2)
            h[-2:] = self.q_tl
        self.f_H = ca.Function('H', [D, P], [H])
        self.f_h = ca.Function('h', [D, P], [h])
        
        # Construct equality constraint matrix and vector
        A_eq = ca.MX.sym('A_eq', ca.Sparsity((self.N+1)*n_z, n_D))
        b_eq = ca.MX.sym('b_eq', ca.Sparsity((self.N+1)*n_z, 1))
        A_eq[:n_z,:n_z] = ca.DM.eye(n_z)
        b_eq[:n_z] = T_xw @ xw0_ph
        for k in range(self.N):
            A_eq[(k+1)*n_z:(k+2)*n_z,k*n_z:(k+1)*n_z] = -A[k]
            A_eq[(k+1)*n_z:(k+2)*n_z,(k+1)*n_z:(k+2)*n_z] = ca.DM.eye(n_z)
            A_eq[(k+1)*n_z:(k+2)*n_z,(self.N+1)*n_z+k*self.n_u:(self.N+1)*n_z+(k+1)*self.n_u] = -B[k]
            b_eq[(k+1)*n_z:(k+2)*n_z] = g[k]
        self.f_A_eq = ca.Function('A_eq', [D, P], [A_eq])
        self.f_b_eq = ca.Function('b_eq', [D, P], [b_eq])

        # Construct inequality constraint matrix and vectors
        n_Cxw = int(np.sum([c.size1() for c in C_xw]))
        n_Cdw = int(np.sum([c.size1() for c in C_dw]))
        A_in = ca.MX.sym('A_in', ca.Sparsity(n_Cxw+n_Cdw, n_D))
        ub_in = ca.MX.sym('ub_in', ca.Sparsity(n_Cxw+n_Cdw, 1))
        lb_in = ca.MX.sym('lb_in', ca.Sparsity(n_Cxw+n_Cdw, 1))
        s1_idx, s2_idx = 0, n_Cxw
        for k in range(self.N+1):
            n_c = C_xw[k].size1()
            A_in[s1_idx:s1_idx+n_c,k*n_z:(k+1)*n_z] = C_xw[k]
            ub_in[s1_idx:s1_idx+n_c] = C_xw_ub[k]
            lb_in[s1_idx:s1_idx+n_c] = C_xw_lb[k]
            if self.soft_track and self.qp_interface in ['casadi']:
                A_in[s1_idx:s1_idx+n_c,-2:] = C_xw_s[k]
            s1_idx += n_c
            if k < self.N:
                n_c = C_dw[k].size1()
                A_in[s2_idx:s2_idx+n_c,(self.N+1)*n_z+k*self.n_u:(self.N+1)*n_z+(k+1)*self.n_u] = C_dw[k]
                ub_in[s2_idx:s2_idx+n_c] = C_dw_ub[k]
                lb_in[s2_idx:s2_idx+n_c] = C_dw_lb[k]
                s2_idx += n_c
        self.f_A_in = ca.Function('A_in', [D, P], [A_in])
        self.f_ub_in = ca.Function('ub_in', [D, P], [ub_in])
        self.f_lb_in = ca.Function('lb_in', [D, P], [lb_in])
        
        # Functions which return the QP components for each stage
        self.f_Qxw = ca.Function('Qxw', [D, P], Q_xw)
        self.f_qxw = ca.Function('qxw', [D, P], q_xw)
        self.f_Qdw = ca.Function('Qdw', [D, P], Q_dw)
        self.f_qdw = ca.Function('qdw', [D, P], q_dw)
        self.f_Cxw = ca.Function('Cxw', [D, P], C_xw)
        self.f_Cxwub = ca.Function('Cxw_ub', [D, P], C_xw_ub)
        self.f_Cxwlb = ca.Function('Cxw_lb', [D, P], C_xw_lb)
        self.f_Cdw = ca.Function('Cdw', [D, P], C_dw)
        self.f_Cdwub = ca.Function('Cdw_ub', [D, P], C_dw_ub)
        self.f_Cdwlb = ca.Function('Cdw_lb', [D, P], C_dw_lb)
        self.f_A = ca.Function('A', [D, P], A)
        self.f_B = ca.Function('B', [D, P], B)
        self.f_g = ca.Function('g', [D, P], g)
        if self.soft_track and self.qp_interface in ['casadi']:
            self.f_Cxws = ca.Function('Cxws', [D, P], C_xw_s)

        if self.qp_interface == 'casadi':
            prob = dict(h=H.sparsity(), a=ca.vertcat(A_eq.sparsity(), A_in.sparsity()))
            # solver = 'qrqp'
            # solver_opts = dict(error_on_fail=False)
            solver = 'osqp'
            solver_opts = dict(error_on_fail=False, osqp={'polish': True, 'verbose': self.verbose})
            self.solver = ca.conic('qp', solver, prob, solver_opts)
        elif self.qp_interface == 'hpipm':
            dims = hp.hpipm_ocp_qp_dim(self.N)
            dims.set('nx', self.n_q+self.n_u, 0, self.N)
            dims.set('nu', self.n_u, 0, self.N-1)
            dims.set('nbx', self.n_q+self.n_u, 0, self.N)
            dims.set('nbu', self.n_u, 0, self.N-1)
            for k in range(self.N+1):
                dims.set('ng', self.n_c[k], k)

                nsg = 0
                if self.soft_track and k >= 1:
                    nsg += 1
                if self.soft_constraint_idxs:
                    nsg += self.n_sc[k]
                dims.set('nsg', nsg, k)

            mode = 'speed'
            arg = hp.hpipm_ocp_qp_solver_arg(dims, mode)
            arg.set('mu0', 1e0)
            arg.set('iter_max', 500)
            arg.set('tol_stat', 1e-6)
            arg.set('tol_eq', 1e-6)
            arg.set('tol_ineq', 1e-6)
            arg.set('tol_comp', 1e-5)
            arg.set('reg_prim', 1e-12)

            self.qp = hp.hpipm_ocp_qp(dims)
            self.sol = hp.hpipm_ocp_qp_sol(dims)
            self.solver = hp.hpipm_ocp_qp_solver(dims, arg)

            # Set up slack variables
            for k in range(self.N+1):
                Jsg = []
                Zu = []
                Zl = []
                zu = []
                zl = []
                lls = []
                lus = []

                if self.soft_track and k >= 1:
                    # Track constraint should be first row
                    J = np.zeros((self.n_c[k], 1))
                    J[0,0] = 1
                    Jsg.append(J)
                    Zu.append(np.array([self.q_tq]))
                    Zl.append(np.array([self.q_tq]))
                    zu.append(np.array([self.q_tl]))
                    zl.append(np.array([self.q_tl]))
                    lls.append(np.array([0]))
                    lus.append(np.array([0]))

                if self.soft_constraint_idxs is not None:
                    if len(self.soft_constraint_idxs[k]) > 0:
                        J = ca.DM.zeros(self.n_c[k], self.n_sc[k])
                        for i, j in enumerate(self.soft_constraint_idxs[k]):
                            J[j+1,i] = 1
                        Jsg.append(J)
                        Zu.append(self.soft_constraint_quad[k])
                        Zl.append(self.soft_constraint_quad[k])
                        zu.append(self.soft_constraint_lin[k])
                        zl.append(self.soft_constraint_lin[k])
                        lls.append(np.zeros(self.n_sc[k]))
                        lus.append(np.zeros(self.n_sc[k]))

                if len(Jsg) > 0:
                    self.qp.set('Jsg', np.hstack(Jsg), k)
                    self.qp.set('Zu', np.concatenate(Zu), k)
                    self.qp.set('Zl', np.concatenate(Zl), k)
                    self.qp.set('zu', np.concatenate(zu).reshape((-1,1)), k)
                    self.qp.set('zl', np.concatenate(zl).reshape((-1,1)), k)
                    self.qp.set('lls', np.concatenate(lls), k)
                    self.qp.set('lus', np.concatenate(lus), k)

        elif self.qp_interface == 'cvxpy':
            pass
        else:
            raise(ValueError('QP interface name not recognized'))

        if self.code_gen:
            # NOTE: Cannot codegen functions which use the CaSAdi matrix exponential function
            generator = ca.CodeGenerator(self.c_file_name)
            generator.add(self.f_H)
            generator.add(self.f_h)
            # generator.add(self.f_A_eq)
            # generator.add(self.f_b_eq)
            generator.add(self.f_A_in)
            generator.add(self.f_ub_in)
            generator.add(self.f_lb_in)

            generator.add(self.f_Qxw)
            generator.add(self.f_qxw)
            generator.add(self.f_Qdw)
            generator.add(self.f_qdw)
            generator.add(self.f_Cxw)
            generator.add(self.f_Cxwub)
            generator.add(self.f_Cxwlb)
            # generator.add(self.f_A)
            # generator.add(self.f_B)
            # generator.add(self.f_g)

            # Set up paths
            cur_dir = pathlib.Path.cwd()
            gen_path = cur_dir.joinpath(self.solver_name)
            c_path = gen_path.joinpath(self.c_file_name)
            if gen_path.exists():
                shutil.rmtree(gen_path)
            gen_path.mkdir(parents=True)

            os.chdir(gen_path)
            if self.verbose:
                self.out(f'- Generating C code for solver {self.solver_name} at {str(gen_path)}')
            generator.generate()
            # Compile into shared object
            so_path = gen_path.joinpath(self.so_file_name)
            command = f'gcc -fPIC -shared -{self.opt_flag} {c_path} -o {so_path}'
            if self.verbose:
                self.out(f'- Compiling shared object {so_path} from {c_path}')
                self.out(f'- Executing "{command}"')
            # pdb.set_trace()
            os.system(command)
            # pdb.set_trace()
            # Swtich back to working directory
            os.chdir(cur_dir)
            install_dir = self.install()

            # Load solver
            self._load_solver(str(install_dir.joinpath(self.so_file_name)))

    def _load_solver(self, solver_path=None):
        # NOTE: Cannot codegen functions which use the CaSAdi matrix exponential function
        if solver_path is None:
            solver_path = str(pathlib.Path(self.solver_dir, self.so_file_name).expanduser())
        if self.verbose:
            self.out(f'- Loading solver from {solver_path}')
        self.f_H = ca.external('H', solver_path)
        self.f_h = ca.external('h', solver_path)
        # self.f_A_eq = ca.external('A_eq', solver_path)
        # self.f_b_eq = ca.external('b_eq', solver_path)
        self.f_A_in = ca.external('A_in', solver_path)
        self.f_ub_in = ca.external('ub_in', solver_path)
        self.f_lb_in = ca.external('lb_in', solver_path)

        self.f_Qxw = ca.external('Qxw', solver_path)
        self.f_qxw = ca.external('qxw', solver_path)
        self.f_Qdw = ca.external('Qdw', solver_path)
        self.f_qdw = ca.external('qdw', solver_path)
        self.f_Cxw = ca.external('Cxw', solver_path)
        self.f_Cxwub = ca.external('Cxw_ub', solver_path)
        self.f_Cxwlb = ca.external('Cxw_lb', solver_path)
        # self.f_A = ca.external('A', solver_path)
        # self.f_B = ca.external('B', solver_path)
        # self.f_g = ca.external('g', solver_path)

    def _solve_casadi(self, D, P):
        if self.verbose:
            self.print_method('============ Sovling using CaSAdi ============')
        t = time.time()

        if self.soft_track:
            D = np.concatenate((D, np.zeros(2)))
            D_scaling = copy.copy(np.concatenate((self.D_scaling, np.ones(2))))
            D_scaling_inv = copy.copy(1/D_scaling)
            ub = copy.copy(np.concatenate((self.ub, np.inf*np.ones(2))))
            lb = copy.copy(np.concatenate((self.lb, np.zeros(2))))
        else:
            D_scaling = copy.copy(self.D_scaling)
            D_scaling_inv = copy.copy(1/D_scaling)
            ub = copy.copy(self.ub)
            lb = copy.copy(self.lb)

        # Evaluate QP approximation
        h       = self.f_h(D, P)
        H       = self.f_H(D, P)
        A_eq    = self.f_A_eq(D, P)
        b_eq    = self.f_b_eq(D, P)
        A_in    = self.f_A_in(D, P)
        ub_in   = self.f_ub_in(D, P)
        lb_in   = self.f_lb_in(D, P)
        if self.verbose:
            self.print_method('Evaluation time: ' + str(time.time()-t))

        lb = D_scaling * lb
        ub = D_scaling * ub

        t = time.time()
        sol = self.solver(h=H, g=h, a=ca.vertcat(A_eq, A_in), lba=ca.vertcat(b_eq, lb_in), uba=ca.vertcat(b_eq, ub_in), lbx=lb, ubx=ub)
        if self.verbose:
            self.print_method('Solve time: ' + str(time.time()-t))

        t = time.time()
        D_bar = None
        success = self.solver.stats()['success']
        status = self.solver.stats()['return_status']

        if success:
            t = time.time()
            D_bar = D_scaling_inv*sol['x'].toarray().squeeze()

            if self.soft_track:
                slacks = D_bar[-2:]
                D_bar = D_bar[:-2]
                if np.amax(slacks) > 1e-9:
                    self.print_method(f'Track constraint violation: {np.amax(slacks)}')
        if self.verbose:
            self.print_method('Unpack time: ' + str(time.time()-t))
            self.print_method('==============================================')

        return D_bar, success, status

    def _solve_cvxpy(self, D, P):
        Q = self.f_Qxw(D, P)
        q = self.f_qxw(D, P)
        R = self.f_Qdw(D, P)
        r = self.f_qdw(D, P)
        Cxw = self.f_Cxw(D, P)
        lbCxw = self.f_Cxwlb(D, P)
        ubCxw = self.f_Cxwub(D, P)
        Cdw = self.f_Cdw(D, P)
        lbCdw = self.f_Cdwlb(D, P)
        ubCdw = self.f_Cdwub(D, P)
        A = self.f_A(D, P)
        B = self.f_B(D, P)
        g = self.f_g(D, P)

        x = cp.Variable(shape=(self.n_q+self.n_u, self.N+1))
        u = cp.Variable(shape=(self.n_u, self.N))
        
        lb = self.xw_scaling * np.concatenate((self.x_lb, self.w_lb))
        ub = self.xw_scaling * np.concatenate((self.x_ub, self.w_ub))

        if self.soft_track:
            s = cp.Variable(shape=(2, 1))

        xw0 = P[:self.n_q+self.n_u]

        constraints, objective = [x[:,0] == self.xw_scaling*xw0], 0
        for k in range(self.N+1):
            objective += 0.5*cp.quad_form(x[:,k], Q[k].full()) + q[k].full().T @ x[:,k]
            if k < self.N:
                objective += 0.5*cp.quad_form(u[:,k], R[k].full()) + r[k].full().T @ u[:,k]
            if k > 0:
                constraints += [x[:,k] <= ub, x[:,k] >= lb]
                if Cxw[k].size1() > 0:
                    if self.soft_track:
                        constraints += [Cxw[k].full()[0] @ x[:,k] - s[0] <= ubCxw[k].full()[0], Cxw[k].full()[0] @ x[:,k] + s[1] >= lbCxw[k].full()[0]]
                        if Cxw[k].size1() > 1:
                            constraints += [Cxw[k].full()[1:] @ x[:,k] <= ubCxw[k].full()[1:], Cxw[k].full()[1:] @ x[:,k] >= lbCxw[k].full()[1:]]
                    else:
                        constraints += [Cxw[k].full() @ x[:,k] <= ubCxw[k].full().squeeze(), Cxw[k].full() @ x[:,k] >= lbCxw[k].full().squeeze()]
            if k < self.N:
                constraints += [x[:,k+1] == A[k].full() @ x[:,k] + B[k].full() @ u[:,k] + g[k].full().squeeze()]
                constraints += [u[:,k] <= self.dw_ub, u[:,k] >= self.dw_lb]
                if Cdw[k].size1() > 0:
                    constraints += [Cdw[k].full() @ x[:,k] <= ubCdw[k].full(), Cdw[k].full() @ x[:,k] >= lbCdw[k].full()]
        if self.soft_track:
            objective += 0.5*cp.quad_form(s, self.q_tq*np.eye(2)) + self.q_tl*np.ones((2,1)).T @ s
            constraints += [s >= 0]

        success = False
        D_bar = None
        prob = cp.Problem(cp.Minimize(objective), constraints)
        # prob.solve(solver='OSQP', verbose=True, eps_abs=1e-6, eps_rel=1e-6, polish=True)
        prob.solve(solver='ECOS', verbose=True)
        # prob.solve(solver='MOSEK', verbose=True)

        if prob.status == 'optimal':
            success = True
            xw_sol, dw_sol = x.value.T @ np.diag(self.xw_scaling_inv), u.value.T
            D_bar = np.concatenate((xw_sol.ravel(), dw_sol.ravel()))
            
            if self.soft_track:
                if np.amax(s.value) > 1e-9:
                    self.print_method(f'Track constraint violation: {np.amax(s.value)}')

        return D_bar, success, prob.status

    def _solve_hpipm(self, D, P):
        env_run = os.getenv('ENV_RUN')
        if env_run!='true':
            self.print_method('ERROR: env.sh has not been sourced! Before executing this example, run:')
            self.print_method('source env.sh')
            sys.exit(1)

        if self.verbose:
            self.print_method('============ Sovling using HPIPM ============')
        t = time.time()
        Q = self.f_Qxw(D, P)
        q = self.f_qxw(D, P)
        R = self.f_Qdw(D, P)
        r = self.f_qdw(D, P)
        Cxw = self.f_Cxw(D, P)
        lbCxw = self.f_Cxwlb(D, P)
        ubCxw = self.f_Cxwub(D, P)
        Cdw = self.f_Cdw(D, P)
        lbCdw = self.f_Cdwlb(D, P)
        ubCdw = self.f_Cdwub(D, P)
        A = self.f_A(D, P)
        B = self.f_B(D, P)
        g = self.f_g(D, P)
        if self.verbose:
            self.print_method('Evaluation time: ' + str(time.time()-t))

        t = time.time()
        lb = self.xw_scaling * np.concatenate((self.x_lb, self.w_lb))
        ub = self.xw_scaling * np.concatenate((self.x_ub, self.w_ub))
        xw0 = P[:self.n_q+self.n_u]
        
        # self.print_method(str((self.xw_scaling*xw0)[6]))
        # self.print_method(str(ub[6]))

        for k in range(self.N+1):
            # Dynamics
            if k < self.N:
                self.qp.set('A', A[k], k)
                self.qp.set('B', B[k], k)
                self.qp.set('b', g[k], k)

            # Cost
            self.qp.set('Q', Q[k], k)
            self.qp.set('q', q[k], k)
            if k < self.N:
                self.qp.set('R', R[k], k)
                self.qp.set('r', r[k], k)

            # Constraints
            if k < self.N and Cdw[k].size1() > 0:
                C = ca.vertcat(Cxw[k], ca.DM.zeros(Cdw[k].size1(), self.n_q+self.n_u))
                D = ca.vertcat(ca.DM.zeros(Cxw[k].size1(), self.n_u, Cdw[k]))
                ug = ca.vertcat(ubCxw[k], ubCdw[k])
                lg = ca.vertcat(lbCxw[k], lbCdw[k])
                self.qp.set('C', C, k)
                self.qp.set('D', D, k)
                self.qp.set('ug', ug, k)
                self.qp.set('lg', lg, k)
            else:
                self.qp.set('C', Cxw[k], k)
                self.qp.set('ug', ubCxw[k], k)
                self.qp.set('lg', lbCxw[k], k)

            # Simple bounds
            self.qp.set('Jbx', np.eye(self.n_q+self.n_u), k)
            if k == 0:
                self.qp.set('ubx', self.xw_scaling*xw0, k)
                self.qp.set('lbx', self.xw_scaling*xw0, k)
            else:
                self.qp.set('ubx', ub, k)
                self.qp.set('lbx', lb, k)
            
            if k < self.N:
                self.qp.set('Jbu', np.eye(self.n_u), k)
                self.qp.set('ubu', self.dw_ub, k)
                self.qp.set('lbu', self.dw_lb, k)

        if self.verbose:
            self.print_method('Construction time: ' + str(time.time()-t))

        t = time.time()
        success = False
        D_bar = None
        self.solver.solve(self.qp, self.sol)
        if self.verbose:
            self.print_method('Solve time: ' + str(time.time()-t))

        t = time.time()
        status = self.solver.get('status')
        # if status == 0:
        if status == 0 or status == 1:
            success = True
            xw, dw = [], []
            for k in range(self.N+1):
                xw.append(self.xw_scaling_inv*self.sol.get('x', k).squeeze())
                if k < self.N:
                    dw.append(self.sol.get('u', k).squeeze())
            D_bar = np.concatenate((np.concatenate(xw),
                                    np.concatenate(dw)))

            # for k in range(self.N+1):
            #     # self.print_method(f'Stage {k}: bound values')
            #     # self.print_method('Upper')
            #     # bu_k = xw[k] - self.xw_scaling_inv*ub
            #     # self.print_method(str(np.where(bu_k > 0)[0]))
            #     # self.print_method('Lower')
            #     # bl_k = self.xw_scaling_inv*lb - xw[k]
            #     # self.print_method(str(np.where(bl_k > 0)[0]))

            #     self.print_method(f'Stage {k}: constraint values')
            #     self.print_method('Upper')
            #     gu_k = Cxw[k] @ (self.xw_scaling*xw[k]) - ubCxw[k]
            #     self.print_method(str(np.where(gu_k > 0)[0]))
            #     self.print_method('Lower')
            #     gl_k = lbCxw[k] - Cxw[k] @ (self.xw_scaling*xw[k])
            #     self.print_method(str(np.where(gl_k > 0)[0]))

            # if self.soft_track:
            #     sl, su = [], []
            #     for k in range(1, self.N+1):
            #         # sl.append(self.sol.get('sl', k).squeeze())
            #         # su.append(self.sol.get('su', k).squeeze())
            #         l = self.sol.get('sl', k)
            #         u = self.sol.get('su', k)
            #         if l.size > 0:
            #             sl.append(l.squeeze())
            #         if u.size > 0:
            #             su.append(u.squeeze())
            #     sl = np.concatenate(sl)
            #     su = np.concatenate(su)
            #     if np.amax(sl) > 1e-9 or np.amax(su) > 1e-9:
            #         vio = max(np.amax(sl), np.amax(su))
            #         self.print_method(f'Track constraint violation: {vio}')
            
            # if self.soft_constraint_idxs is not None:
            #     sl, su = [], []
            #     for k in range(self.N+1):
            #         # sl.append(self.sol.get('sl', k).squeeze())
            #         # su.append(self.sol.get('su', k).squeeze())
            #         l = self.sol.get('sl', k)
            #         u = self.sol.get('su', k)
            #         if l.size > 0:
            #             sl.append(l.squeeze()[1:])
            #         if u.size > 0:
            #             su.append(u.squeeze()[1:])
            #     sl = np.concatenate(sl)
            #     su = np.concatenate(su)
            #     if np.amax(sl) > 1e-9 or np.amax(su) > 1e-9:
            #         vio = max(np.amax(sl), np.amax(su))
            #         self.print_method(f'Soft constraint violation: {vio}')

        if self.verbose:
            self.print_method('Unpack time: ' + str(time.time()-t))
            self.print_method('=============================================')

        return D_bar, success, status

if __name__ == "__main__":
    pass