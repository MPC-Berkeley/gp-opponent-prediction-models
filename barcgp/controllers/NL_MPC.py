#!/usr/bin python3

import pdb
import array
import warnings

import numpy as np

import casadi as ca

import sys, os, pathlib

sys.path.append(os.path.join(os.path.expanduser('~'), 'forces_pro_client'))
import forcespro
import forcespro.nlp

from barcgp.dynamics.models.dynamics_models import CasadiDynamicsModel

from barcgp.common.pytypes import VehicleState, VehicleActuation, VehiclePrediction
from barcgp.common.utils.file_utils import *

from barcgp.controllers.abstract_controller import AbstractController
from barcgp.controllers.utils.controllerTypes import NLMPCParams


class NL_MPC(AbstractController):
    def __init__(self, dynamics, racetrack, control_params=NLMPCParams()):

        assert isinstance(dynamics, CasadiDynamicsModel)
        self.dynamics = dynamics

        self.track = racetrack

        self.dt = control_params.dt
        self.n = control_params.n
        self.d = control_params.d

        self.N = control_params.N

        self.Q = control_params.Q
        self.R = control_params.R
        self.Q_f = control_params.Q_f
        self.R_d = control_params.R_d
        self.Q_s = control_params.Q_s
        self.l_s = control_params.l_s

        self.state_ub = control_params.state_ub
        self.state_lb = control_params.state_lb
        self.input_ub = control_params.input_ub
        self.input_lb = control_params.input_lb
        self.input_rate_ub = control_params.input_rate_ub
        self.input_rate_lb = control_params.input_rate_lb

        self.optlevel = control_params.optlevel
        self.slack = control_params.slack
        self.solver_dir = control_params.solver_dir
        self.install_dir = None

        self.u_prev = np.zeros(self.d)
        self.x_pred = np.zeros((self.N + 1, self.n))
        self.u_pred = np.zeros((self.N, self.d))
        self.x_ws = None
        self.u_ws = None
        self.curvatures = None
        self.halfwidths = None

        self.model = None
        self.options = None
        self.solver = None
        self.solver_name = 'NL_MPC_solver_forces_pro'

        self.state_input_prediction = VehiclePrediction()

        self.initialized = False

    def initialize(self):
        if self.solver_dir:
            self.solver_dir = pathlib.Path(self.solver_dir).expanduser()  # allow the use of ~
            self._load_solver(self.solver_dir)
        else:
            self._build_solver()

        self.initialized = True

    def step(self, vehicle_state, env_state = None):
        info = self.solve(vehicle_state, self.x_ref)

        self.dynamics.qu2state(vehicle_state, None, self.u_pred[0])
        self.dynamics.qu2prediction(self.state_input_prediction, self.x_pred, self.u_pred)
        self.state_input_prediction.t = vehicle_state.t

        return info

    def get_prediction(self):
        return self.state_input_prediction

    def get_ss(self):
        return VehiclePrediction()

    def set_warm_start(self, x_ws, u_ws):
        if x_ws.shape[0] != self.N + 1 or x_ws.shape[1] != self.n:
            raise (RuntimeError(
                'Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    x_ws.shape[0], x_ws.shape[1], self.N + 1, self.n)))
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.d:
            raise (RuntimeError(
                'Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    u_ws.shape[0], u_ws.shape[1], self.N, self.d)))

        self.x_ws = x_ws
        self.u_ws = u_ws

        self.halfwidths = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            self.halfwidths[i] = self.track.get_halfwidth(self.x_ws[i, 4])

    def set_x_ref(self, x_ref):
         self.x_ref = x_ref

    def solve(self, state, x_ref = None, input_prev=None):
        if not self.initialized:
            raise (RuntimeError(
                'NL MPC controller is not initialized, run NL_MPC.initialize() before calling NL_MPC.solve()'))

        x, _ = self.dynamics.state2qu(state)
        if input_prev is not None:
            self.u_prev = self.dynamics.input2u(input_prev)

        if self.x_ws is None:
            warnings.warn('Initial guess of open loop state sequence not provided, using zeros')
            self.x_ws = np.zeros((self.N + 1, self.n))
            self.halfwidths = np.zeros(self.N + 1)
            for i in range(self.N + 1):
                self.halfwidths[i] = self.track.get_halfwidth(self.x_ws[i, 4])
        if self.u_ws is None:
            warnings.warn('Initial guess of open loop input sequence not provided, using zeros')
            self.u_ws = np.zeros((self.N, self.d))


        if x_ref is not None:
            self.x_ref = x_ref

        parameters = []; initial_guess = []

        for i in range(self.N):
            initial_guess.append(self.x_ws[i])
            initial_guess.append(self.u_ws[i])
            initial_guess.append(self.u_ws[i - 1])

            parameters.append(x_ref[i])
            parameters.append(self.Q)
            parameters.append(self.R)
            parameters.append(self.R_d)

            if self.slack:
                initial_guess.append(np.zeros((1,)))
                parameters.append(np.array(self.Q_s).reshape((-1)))
                parameters.append(np.array(self.l_s).reshape((-1)))
                parameters.append(np.array(self.halfwidths[-1]).reshape((-1)))

        parameters.append(x_ref[-1])
        parameters.append(self.Q_f)

        initial_guess.append(self.x_ws[-1])
        if self.slack:
            initial_guess.append(np.zeros((1,)))
            parameters.append(np.array(self.Q_s).reshape((-1)))
            parameters.append(np.array(self.l_s).reshape((-1)))
            parameters.append(np.array(self.halfwidths[-1]).reshape((-1)))

        initial_guess = np.concatenate(initial_guess)
        parameters = np.concatenate(parameters)

        problem = dict()
        problem['xinit'] = np.concatenate((x, self.u_prev))
        problem['all_parameters'] = parameters
        problem['x0'] = initial_guess

        output, exitflag, solve_info = self.solver.solve(problem)

        if exitflag == 1:
            info = {"success": True, "return_status": "Successfully Solved", "solve_time": solve_info.solvetime,
                    "info": solve_info}
            for k in range(self.N):
                sol = output["x%02d" % (k + 1)]
                self.x_pred[k, :] = sol[:self.n]
                self.u_pred[k, :] = sol[self.n:self.n + self.d]
            sol = output["x%02d" % (self.N + 1)]
            self.x_pred[self.N, :] = sol[:self.n]

            # Construct initial guess for next iteration
            x_ws = self.x_pred[1:]
            u_ws = self.u_pred[1:]
            x_ws = np.vstack((x_ws, np.array(self.dynamics.f_d_rk4(x_ws[-1], u_ws[-1])).squeeze()))
            u_ws = np.vstack((u_ws, u_ws[-1]))
            self.set_warm_start(x_ws, u_ws)

        else:
            info = {"success": False, "return_status": 'Solving Failed, exitflag = %d' % exitflag, "solve_time": None,
                    "info": solve_info}

        self.u_prev = self.u_pred[0]

        return info

    def _load_solver(self, solver_dir):
        if not dir_exists(solver_dir):
            print("====================")
            print("Solver files do not exist, rebuilding and saving config file to", solver_dir, "...")
            print("====================")
            solver_found = False
        else:
            solver_found = True
        if solver_found:
            print('Found solver files for solver', self.solver_name)
        else:
            self._build_solver()
        self.solver = forcespro.nlp.Solver.from_directory(solver_dir)

    def _build_solver(self):
        def stage_quadratic_cost(z, p):
            q = z[:self.n]
            u = z[self.n:self.n + self.d]
            u_prev = z[self.n + self.d:self.n + self.d + self.d]

            q_ref = p[:self.n]
            Q = ca.diag(p[self.n:self.n + self.n])
            R = ca.diag(p[self.n + self.n:self.n + self.n + self.d])
            R_d = ca.diag(p[self.n + self.n + self.d:self.n + self.n + self.d + self.d])

            return ca.bilin(Q, q - q_ref, q - q_ref) \
                   + ca.bilin(R, u, u) \
                   + ca.bilin(R_d, u - u_prev, u - u_prev)

        def stage_slack_cost(z, p):
            slack = z[self.n + self.d + self.d]

            Q_s = p[self.n + self.n + self.d + self.d]
            l_s = p[self.n + self.n + self.d + self.d + 1]

            return Q_s * ca.power(slack, 2) + l_s * slack

        def terminal_quadratic_cost(z, p):
            q = z[:self.n]

            q_ref = p[:self.n]
            Q = ca.diag(p[self.n:self.n + self.n])

            return ca.bilin(Q, q - q_ref, q - q_ref)

        def terminal_slack_cost(z, p):
            slack = z[self.n]

            Q_s = p[self.n + self.n]
            l_s = p[self.n + self.n + 1]

            return Q_s * ca.power(slack, 2) + l_s * slack

        def stage_equality_constraint(z, p):
            q = z[:self.n]
            u = z[self.n:self.n + self.d]

            return ca.vertcat(self.dynamics.f_d_rk4(q, u), u)

        def terminal_equality_constraint(z, p):
            q = z[:self.n]
            u = z[self.n:self.n + self.d]

            return self.dynamics.f_d_rk4(q, u)

        def stage_input_rate_constraint(z, p):
            u = z[self.n:self.n + self.d]
            u_prev = z[self.n + self.d:self.n + self.d + self.d]

            return u - u_prev

        def stage_soft_lane_constraint(z, p):
            e_y = z[5]
            slack = z[self.n + self.d + self.d]

            half_width = p[self.n + self.n + self.d + self.d + 1 + 1]

            return ca.vertcat(e_y - slack - half_width, e_y + slack + half_width)

        def terminal_soft_lane_constraint(z, p):
            e_y = z[5]
            slack = z[self.n]

            half_width = p[self.n + self.n + 1 + 1]

            return ca.vertcat(e_y - slack - half_width, e_y + slack + half_width)

        # Build solver
        self.model = forcespro.nlp.SymbolicModel(self.N + 1)

        # Objective function
        if self.slack:
            stage_objective_function = lambda z, p: stage_quadratic_cost(z, p) + stage_slack_cost(z, p)
            terminal_objective_function = lambda z, p: terminal_quadratic_cost(z, p) + terminal_slack_cost(z, p)
        else:
            stage_objective_function = lambda z, p: stage_quadratic_cost(z, p)
            terminal_objective_function = lambda z, p: terminal_quadratic_cost(z, p)

        # Number of decision variables
        stage_n_var = self.n + self.d + self.d  # q, u, u_prev
        terminal_n_var = self.n  # q
        if self.slack:
            stage_n_var += 1  # lane slack
            terminal_n_var += 1

        # Number of parameters
        stage_n_par = self.n + self.n + self.d + self.d  # q_ref, diag(Q), diag(R), diag(R_d)
        terminal_n_par = self.n + self.n  # q_ref, diag(Q)
        if self.slack:
            stage_n_par += 1 + 1 + 1  # quad slack, lin slack, half width
            terminal_n_par += 1 + 1 + 1  # quad slack, lin slack, half width

        # Equality constraints in the form
        # E_k*z_kp1 = f(z_k, p_k)
        # Where E_k is a selection matrix of dimension n_eq x n_var_kp1
        stage_n_eq = self.n + self.d
        stage_E_q = np.hstack((np.eye(self.n), np.zeros((self.n, self.d + self.d))))
        stage_E_u = np.hstack((np.zeros((self.d, self.n + self.d)), np.eye(self.d)))
        stage_E = np.vstack((stage_E_q, stage_E_u))
        if self.slack:
            stage_E = np.hstack((stage_E, np.zeros((stage_E.shape[0], 1))))

        terminal_n_eq = self.n
        terminal_E = np.eye(self.n)
        if self.slack:
            terminal_E = np.hstack((terminal_E, np.zeros((terminal_E.shape[0], 1))))

        # Nonlinear inequality constraints
        stage_n_h = self.d  # d_u
        stage_hu = self.input_rate_ub * self.dt
        stage_hl = self.input_rate_lb * self.dt
        if self.slack:
            stage_n_h += 2  # soft lane boundary constraints
            stage_hu = np.concatenate((stage_hu, np.array([0, np.inf])))
            stage_hl = np.concatenate((stage_hl, np.array([-np.inf, 0])))
            stage_inequality_constraints = lambda z, p: ca.vertcat(stage_input_rate_constraint(z, p),
                                                                   stage_soft_lane_constraint(z, p))
        else:
            stage_inequality_constraints = lambda z, p: stage_input_rate_constraint(z, p)

        terminal_n_h = 0
        terminal_hu = []
        terminal_hl = []
        terminal_inequality_constraints = None
        if self.slack:
            terminal_n_h += 2  # soft lane boundary constraints
            terminal_hu = np.concatenate((terminal_hu, np.array([0, np.inf])))
            terminal_hl = np.concatenate((terminal_hl, np.array([-np.inf, 0])))
            terminal_inequality_constraints = lambda z, p: terminal_soft_lane_constraint(z, p)

        # Simple upper and lower bounds on decision vector
        stage_ub = np.concatenate((self.state_ub, self.input_ub, self.input_ub))
        stage_lb = np.concatenate((self.state_lb, self.input_lb, self.input_lb))
        if self.slack:
            # Remove hard constraints on lane boundaries
            stage_ub[5] = np.inf;
            stage_lb[5] = -np.inf
            # Positivity constraint on slack variable
            stage_ub = np.append(stage_ub, np.inf)
            stage_lb = np.append(stage_lb, 0)

        terminal_ub = self.state_ub
        terminal_lb = self.state_lb
        if self.slack:
            # Remove hard constraints on lane boundaries
            terminal_ub[5] = np.inf;
            terminal_lb[5] = -np.inf
            # Positivity constraint on slack variable
            terminal_ub = np.append(terminal_ub, np.inf)
            terminal_lb = np.append(terminal_lb, 0)

        for i in range(self.N):
            self.model.nvar[i] = stage_n_var
            self.model.npar[i] = stage_n_par

            self.model.objective[i] = stage_objective_function
            self.model.nh[i] = stage_n_h
            self.model.ineq[i] = stage_inequality_constraints
            self.model.hu[i] = stage_hu
            self.model.hl[i] = stage_hl

            self.model.ub[i] = stage_ub
            self.model.lb[i] = stage_lb

            if i < self.N - 1:
                self.model.neq[i] = stage_n_eq
                self.model.eq[i] = stage_equality_constraint
                self.model.E[i] = stage_E
            else:
                self.model.neq[-1] = terminal_n_eq
                self.model.eq[-1] = terminal_equality_constraint
                self.model.E[-1] = terminal_E

        self.model.nvar[-1] = terminal_n_var
        self.model.npar[-1] = terminal_n_par

        self.model.objective[-1] = terminal_objective_function
        self.model.nh[-1] = terminal_n_h
        self.model.ineq[-1] = terminal_inequality_constraints
        self.model.hu[-1] = terminal_hu
        self.model.hl[-1] = terminal_hl

        self.model.ub[-1] = terminal_ub
        self.model.lb[-1] = terminal_lb

        # Initial conditions
        self.model.xinitidx = list(range(self.n)) + list(range(self.n + self.d, self.n + self.d + self.d))

        # Define solver options
        self.options = forcespro.CodeOptions(self.solver_name)

        self.options.overwrite = True
        self.options.printlevel = 0
        self.options.optlevel = self.optlevel
        self.options.BuildSimulinkBlock = False
        self.options.cleanup = True
        self.options.platform = 'Generic'
        self.options.gnu = True
        self.options.sse = True
        self.options.noVariableElimination = True

        self.options.nlp.linear_solver = 'normal_eqs'
        self.options.nlp.TolStat = 1e-4
        self.options.nlp.TolEq = 1e-4
        self.options.nlp.TolIneq = 1e-4

        # Generate solver (this may take a while for high optlevel)
        self.model.generate_solver(self.options)
        self.install_dir = self.install()  # install the model to ~/.mpclab_controllers
        self.solver = forcespro.nlp.Solver.from_directory(self.install_dir)