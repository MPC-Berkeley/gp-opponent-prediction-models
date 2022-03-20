#!/usr/bin python3

import numpy as np
import scipy.linalg as la
import scipy.signal

import casadi as ca

from abc import abstractmethod
import array
from typing import Tuple, List

import pdb

from barcgp.common.pytypes import VehicleState, VehicleActuation, VehiclePrediction
from barcgp.dynamics.models.model_types import DynamicsConfig, DynamicBicycleConfig
from barcgp.dynamics.models.abstract_model import AbstractModel
from barcgp.common.tracks.track import get_track


class CasadiDynamicsModel(AbstractModel):
    '''
    Base class for dynamics models that use casadi for their models.
    Implements common functions for linearizing models, integrating models, etc...
    '''

    # whether or not the dynamics depend on curvature
    # curvature is not a state variable as it must be calculated from the track definition
    # so any such model has a third input that is needed for calculations!
    curvature_model = False

    def __init__(self, t0: float, model_config: DynamicsConfig, track=None):
        super().__init__(model_config)

        self.t0 = t0
        # Try to Load track by name first, then by track object passed into model init
        if model_config.track_name is not None:
            self.track = get_track(model_config.track_name)
        else:
            self.track = track

        self.dt = model_config.dt
        self.M = model_config.M # RK4 integration steps
        self.h = self.dt/self.M # RK4 integration time intervals

    @abstractmethod
    def state2qu(self):
        pass

    @abstractmethod
    def state2q(self):
        pass

    @abstractmethod
    def input2u(self):
        pass

    @abstractmethod
    def q2state(self):
        pass

    def precompute_model(self):
        '''
        wraps up model initialization
        require the following fields to be initialized:
        self.sym_q:  ca.SX with elements of state vector q
        self.sym_u:  ca.SX with elements of control vector u
        self.sym_dq: ca.SX with time derivatives of q (dq/dt = sym_dq(q,u))
        '''

        dyn_inputs = [self.sym_q, self.sym_u]

        # Continuous time dynamics function
        self.fc = ca.Function('fc', dyn_inputs, [self.sym_dq], self.options('fc'))

        # First derivatives
        self.sym_Ac = ca.jacobian(self.sym_dq, self.sym_q)
        self.sym_Bc = ca.jacobian(self.sym_dq, self.sym_u)
        self.sym_Cc = self.sym_dq

        self.fA = ca.Function('fA', dyn_inputs, [self.sym_Ac], self.options('fA'))
        self.fB = ca.Function('fB', dyn_inputs, [self.sym_Bc], self.options('fB'))
        self.fC = ca.Function('fC', dyn_inputs, [self.sym_Cc], self.options('fC'))

        # Discretization
        if self.model_config.discretization_method == 'euler':
            sym_q_kp1 = self.sym_q + self.dt * self.fc(*dyn_inputs)
        elif self.model_config.discretization_method == 'rk4':
            sym_q_kp1 = self.f_d_rk4(*dyn_inputs)
        else:
            raise ValueError('Discretization method of %s not recognized' % self.model_config.discretization_method)

        if self.model_config.noise:
            if self.model_config.noise_cov is None:
                raise RuntimeError('Noise covariance matrix not provided to dynamics model')
            noise_cov = np.array(self.model_config.noise_cov)
            self.noise_cov = np.diag(noise_cov) if noise_cov.ndim == 1 else noise_cov

            # Symbolic variables for additive process noise components for each state
            self.n_m = self.n_q
            self.sym_m = ca.SX.sym('m', self.n_m)
            sym_q_kp1 += ca.mtimes(la.sqrtm(self.noise_cov), self.sym_m)
            dyn_inputs += [self.sym_m]

        # Discrete time dynamics function
        self.fd = ca.Function('fd', dyn_inputs, [sym_q_kp1], self.options('fd'))

        # First derivatives
        self.sym_Ad = ca.jacobian(sym_q_kp1, self.sym_q)
        self.sym_Bd = ca.jacobian(sym_q_kp1, self.sym_u)
        self.sym_Cd = sym_q_kp1

        self.fAd = ca.Function('fAd', dyn_inputs, [self.sym_Ad], self.options('fAd'))
        self.fBd = ca.Function('fBd', dyn_inputs, [self.sym_Bd], self.options('fBd'))
        self.fCd = ca.Function('fCd', dyn_inputs, [self.sym_Cd], self.options('fCd'))

        # Second derivatives
        self.sym_Ed = [ca.hessian(sym_q_kp1[i], self.sym_q)[0] for i in range(self.n_q)]
        self.sym_Fd = [ca.hessian(sym_q_kp1[i], self.sym_u)[0] for i in range(self.n_q)]
        self.sym_Gd = [ca.jacobian(ca.jacobian(sym_q_kp1[i], self.sym_u), self.sym_q) for i in range(self.n_q)]
        
        self.fEd = ca.Function('fEd', dyn_inputs, self.sym_Ed, self.options('fEd'))
        self.fFd = ca.Function('fFd', dyn_inputs, self.sym_Fd, self.options('fFd'))
        self.fGd = ca.Function('fGd', dyn_inputs, self.sym_Gd, self.options('fGd'))
        
        if self.model_config.noise:
            self.sym_Md = ca.jacobian(sym_q_kp1, self.sym_m)
            self.fMd = ca.Function('fMd', dyn_inputs, [self.sym_Md], self.options('fMd'))

        return

    def local_discretization(self, vehicle_state: VehicleState, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Local Euler discretization of continuous time dynamics
        x_{k+1} = A x_k + B u_k + C
        '''
        q, u = self.state2qu(vehicle_state)  # TODO state2qu
        args = [q, u]

        A = np.eye(q.shape[0]) + dt * np.array(self.fA(*args))
        B = dt * np.array(self.fB(*args))
        C = q + dt * np.array(self.fC(*args)).squeeze() - A @ q - B @ u

        if C.ndim == 1:
            C = np.expand_dims(C, 1)
        return A, B, C

    def step(self, vehicle_state: VehicleState,
             method: str = 'RK45'):
        '''
        steps noise-free model forward one time step (self.dt) using numerical integration
        '''
        q, u = self.state2qu(vehicle_state)  # TODO state2qu

        f = lambda t, qs: (self.fc(qs, u)).toarray().squeeze()

        t = vehicle_state.t - self.t0
        tf = t + self.dt
        q_n = np.array(self.f_d_rk4(q, u))[:,0]
        self.qu2state(vehicle_state, q_n, u)  # TODO qu2state
        vehicle_state.t = tf + self.t0
        vehicle_state.a.a_long, vehicle_state.a.a_tran = self.faccel(q_n, u)

        if self.curvature_model:
            self.track.local_to_global_typed(vehicle_state)
        else:
            self.track.global_to_local_typed(vehicle_state)
        return

    def f_d_rk4(self, x, u):
        '''
        Discrete nonlinear dynamics (RK4 approx.)
        '''
        x_p = x
        for i in range(self.M):
            a1 = self.fc(x_p, u)
            a2 = self.fc(x_p + (self.h / 2) * a1, u)
            a3 = self.fc(x_p + (self.h / 2) * a2, u)
            a4 = self.fc(x_p + self.h * a3, u)
            x_p += self.h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        return x_p


class CasadiDynamicBicycleFull(CasadiDynamicsModel):
    '''
    Global frame of reference dynamic bicycle model - Pacejka model tire forces

    Body frame velocities and global frame positions
    '''

    def __init__(self, t0: float,
                    model_config: DynamicBicycleConfig = DynamicBicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = True
        self.full_model = True
        self.get_curvature = self.track.get_curvature_casadi_fn()
        self.get_curvature_f = self.track.get_curvature_casadi_fn_dynamic()

        self.n_q = 9
        self.n_u = 2

        self.L_f = self.model_config.wheel_dist_front
        self.L_r = self.model_config.wheel_dist_rear
        self.m = self.model_config.mass
        self.I_z = self.model_config.yaw_inertia
        self.c_f = self.model_config.drag_coefficient
        self.pacejka_B = self.model_config.pacejka_b
        self.pacejka_C = self.model_config.pacejka_c
        self.pacejka_Df = self.model_config.pacejka_d_front
        self.pacejka_Dr = self.model_config.pacejka_d_rear
        self.mu = self.model_config.wheel_friction
        self.g = self.model_config.gravity

        # symbolic variables
        self.sym_vx = ca.SX.sym(
            'vx')  # body fram vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_vy = ca.SX.sym('vy')
        self.sym_psidot = ca.SX.sym('psidot')

        # Global states
        self.sym_psi = ca.SX.sym('psi')
        self.sym_x = ca.SX.sym('x')
        self.sym_y = ca.SX.sym('y')

        # Local states
        self.sym_epsi = ca.SX.sym('epsi')
        self.sym_s = ca.SX.sym('s')
        self.sym_xtran = ca.SX.sym('xtran')

        # Track parameters
        self.track_length = ca.SX.sym('track_length')
        self.key_pts = ca.SX.sym('key_pts', 5, 6)

        self.sym_u_s = ca.SX.sym('gamma')
        self.sym_u_a = ca.SX.sym('a')

        self.sym_c = self.get_curvature(self.sym_s)
        self.sym_c_f = self.get_curvature_f(self.sym_s, self.track_length, self.key_pts)


        # Slip angles and Pacejka tire forces
        self.sym_alpha_f = -ca.atan2(
            (self.sym_vy + self.L_f * self.sym_psidot) * ca.cos(self.sym_u_s) - self.sym_vx * ca.sin(self.sym_u_s),
            self.sym_vx * ca.cos(self.sym_u_s) + (self.sym_vy + self.L_f * self.sym_psidot) * ca.sin(self.sym_u_s))
        self.sym_alpha_r = -ca.atan2(self.sym_vy - self.L_r * self.sym_psidot, self.sym_vx)

        self.sym_fyf = self.pacejka_Df * ca.sin(self.pacejka_C * ca.atan(self.pacejka_B * self.sym_alpha_f))
        self.sym_fyr = self.pacejka_Dr * ca.sin(self.pacejka_C * ca.atan(self.pacejka_B * self.sym_alpha_r))

        # instantaneous accelerations
        self.sym_ax = self.sym_u_a - self.c_f * self.sym_vx - self.sym_fyf * ca.sin(self.sym_u_s) / self.m
        self.sym_ay = (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m
        self.sym_wz = (self.L_f * self.sym_fyf * ca.cos(self.sym_u_s) - self.L_r * self.sym_fyr) / self.I_z

        # time derivatives
        self.sym_dvx = self.sym_ax + self.sym_psidot * self.sym_vy
        self.sym_dvy = self.sym_ay - self.sym_psidot * self.sym_vx
        self.sym_dpsidot = self.sym_wz

        self.sym_dx = self.sym_vx * ca.cos(self.sym_psi) - self.sym_vy * ca.sin(self.sym_psi)
        self.sym_dy = self.sym_vy * ca.cos(self.sym_psi) + self.sym_vx * ca.sin(self.sym_psi)
        self.sym_dpsi = self.sym_psidot

        self.sym_depsi = self.sym_psidot - self.sym_c * (self.sym_vx * ca.cos(self.sym_epsi) - self.sym_vy * ca.sin(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c)
        self.sym_ds = (self.sym_vx * ca.cos(self.sym_epsi) - self.sym_vy * ca.sin(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c)
        self.sym_depsi_f = self.sym_psidot - self.sym_c_f * (self.sym_vx * ca.cos(self.sym_epsi) - self.sym_vy * ca.sin(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c_f)
        self.sym_ds_f = (self.sym_vx * ca.cos(self.sym_epsi) - self.sym_vy * ca.sin(self.sym_epsi)) / (
                    1 - self.sym_xtran * self.sym_c_f)
        self.sym_dxtran = self.sym_vx * ca.sin(self.sym_epsi) + self.sym_vy * ca.cos(self.sym_epsi)

        # state and state derivative functions
        self.sym_q = ca.vertcat(self.sym_vx, self.sym_vy, self.sym_psidot, self.sym_x, self.sym_y, self.sym_psi,
                                self.sym_epsi, self.sym_s, self.sym_xtran)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_dvx, self.sym_dvy, self.sym_dpsidot, self.sym_dx, self.sym_dy, self.sym_dpsi,
                                 self.sym_depsi, self.sym_ds, self.sym_dxtran)

        self.sym_dq_f = ca.vertcat(self.sym_dvx, self.sym_dvy, self.sym_dpsidot, self.sym_dx, self.sym_dy, self.sym_dpsi,
                                 self.sym_depsi_f, self.sym_ds_f, self.sym_dxtran)

        self.faccel = ca.Function('faccel', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay], self.options('faccel'))

        self.precompute_model()
        self.precompute_model_forces()
        return


    def f_d_rk4_forces(self, x, track_length, key_pts, u):
        '''
        Discrete nonlinear dynamics (RK4 approx.)
        '''
        x_p = x
        for i in range(self.M):
            a1 = self.fc_f(x_p, track_length, key_pts, u)
            a2 = self.fc_f(x_p + (self.h / 2) * a1, track_length, key_pts, u)
            a3 = self.fc_f(x_p + (self.h / 2) * a2, track_length, key_pts, u)
            a4 = self.fc_f(x_p + self.h * a3, track_length, key_pts, u)
            x_p += self.h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        return x_p

    def precompute_model_forces(self):
        '''
        wraps up model initialization
        require the following fields to be initialized:
        self.sym_q:  ca.SX with elements of state vector q
        self.sym_u:  ca.SX with elements of control vector u
        self.key_pts:  ca.SX with 4 key points
        self.track_length:  ca.SX track length
        self.sym_dq: ca.SX with time derivatives of q (dq/dt = sym_dq(q,u))
        '''

        dyn_inputs = [self.sym_q, self.track_length, self.key_pts, self.sym_u]

        # Continuous time dynamics function
        self.fc_f = ca.Function('fc', dyn_inputs, [self.sym_dq_f], self.options('fc'))

        # First derivatives
        self.sym_Ac_f = ca.jacobian(self.sym_dq_f, self.sym_q)
        self.sym_Bc_f = ca.jacobian(self.sym_dq_f, self.sym_u)
        self.sym_Cc_f = self.sym_dq_f

        self.fA_f = ca.Function('fA', dyn_inputs, [self.sym_Ac_f], self.options('fA'))
        self.fB_f = ca.Function('fB', dyn_inputs, [self.sym_Bc_f], self.options('fB'))
        self.fC_f = ca.Function('fC', dyn_inputs, [self.sym_Cc_f], self.options('fC'))

        # Discretization
        if self.model_config.discretization_method == 'euler':
            sym_q_kp1 = self.sym_q + self.dt * self.fc_f(*dyn_inputs)
        elif self.model_config.discretization_method == 'rk4':
            sym_q_kp1 = self.f_d_rk4_forces(*dyn_inputs)
        else:
            raise ValueError('Discretization method of %s not recognized' % self.model_config.discretization_method)

        if self.model_config.noise:
            if self.model_config.noise_cov is None:
                raise RuntimeError('Noise covariance matrix not provided to dynamics model')
            noise_cov = np.array(self.model_config.noise_cov)
            self.noise_cov = np.diag(noise_cov) if noise_cov.ndim == 1 else noise_cov

            # Symbolic variables for additive process noise components for each state
            self.n_m = self.n_q
            self.sym_m = ca.SX.sym('m', self.n_m)
            sym_q_kp1 += ca.mtimes(la.sqrtm(self.noise_cov), self.sym_m)
            dyn_inputs += [self.sym_m]

        # Discrete time dynamics function
        self.fd_f = ca.Function('fd', dyn_inputs, [sym_q_kp1], self.options('fd'))

        # First derivatives
        self.sym_Ad_f = ca.jacobian(sym_q_kp1, self.sym_q)
        self.sym_Bd_f = ca.jacobian(sym_q_kp1, self.sym_u)
        self.sym_Cd_f = sym_q_kp1

        self.fAd_f = ca.Function('fAd', dyn_inputs, [self.sym_Ad_f], self.options('fAd'))
        self.fBd_f = ca.Function('fBd', dyn_inputs, [self.sym_Bd_f], self.options('fBd'))
        self.fCd_f = ca.Function('fCd', dyn_inputs, [self.sym_Cd_f], self.options('fCd'))

        # Second derivatives
        self.sym_Ed_f = [ca.hessian(sym_q_kp1[i], self.sym_q)[0] for i in range(self.n_q)]
        self.sym_Fd_f = [ca.hessian(sym_q_kp1[i], self.sym_u)[0] for i in range(self.n_q)]
        self.sym_Gd_f = [ca.jacobian(ca.jacobian(sym_q_kp1[i], self.sym_u), self.sym_q) for i in range(self.n_q)]

        self.fEd_f = ca.Function('fEd', dyn_inputs, self.sym_Ed_f, self.options('fEd'))
        self.fFd_f = ca.Function('fFd', dyn_inputs, self.sym_Fd_f, self.options('fFd'))
        self.fGd_f = ca.Function('fGd', dyn_inputs, self.sym_Gd_f, self.options('fGd'))

        if self.model_config.noise:
            self.sym_Md = ca.jacobian(sym_q_kp1, self.sym_m)
            self.fMd = ca.Function('fMd', dyn_inputs, [self.sym_Md], self.options('fMd'))

        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi, state.p.e_psi,
                      state.p.s, state.p.x_tran])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.x.x, state.x.y, state.e.psi, state.p.e_psi,
                      state.p.s, state.p.x_tran])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.v.v_long = q[0]
        state.v.v_tran = q[1]
        state.w.w_psi = q[2]
        state.x.x = q[3]
        state.x.y = q[4]
        state.e.psi = q[5]
        state.p.e_psi = q[6]
        state.p.s = q[7]
        state.p.x_tran = q[8]
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a = u[0]
        input.u_steer = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.v.v_long = q[0]
            state.v.v_tran = q[1]
            state.w.w_psi = q[2]
            state.x.x = q[3]
            state.x.y = q[4]
            state.e.psi = q[5]
            state.p.e_psi = q[6]
            state.p.s = q[7]
            state.p.x_tran = q[8]
        if u is not None:
            state.u.u_a = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            prediction.v_long = array.array('d', q[:, 0])
            prediction.v_tran = array.array('d', q[:, 1])
            prediction.psidot = array.array('d', q[:, 2])
            prediction.x = array.array('d', q[:, 3])
            prediction.y = array.array('d', q[:, 4])
            prediction.psi = array.array('d', q[:, 5])
            prediction.e_psi = array.array('d', q[:, 6])
            prediction.s = array.array('d', q[:, 7])
            prediction.x_tran = array.array('d', q[:, 8])
        if u is not None:
            prediction.u_a = array.array('d', u[:, 0])
            prediction.u_steer = array.array('d', u[:, 1])

class CasadiDynamicCLBicycle(CasadiDynamicsModel):
    '''
    Frenet frame of reference dynamic bicycle model - Pacejka model tire forces

    Body frame velocities and track frame positions
    '''
    def __init__(self, t0: float, model_config: DynamicBicycleConfig = DynamicBicycleConfig(), track=None):
        super().__init__(t0, model_config, track=track)

        self.curvature_model = True
        self.get_curvature = self.track.get_curvature_casadi_fn()

        self.n_q = 6
        self.n_u = 2

        self.L_f = self.model_config.wheel_dist_front
        self.L_r = self.model_config.wheel_dist_rear
        self.m = self.model_config.mass
        self.I_z = self.model_config.yaw_inertia
        self.c_f = self.model_config.drag_coefficient
        self.pacejka_B = self.model_config.pacejka_b
        self.pacejka_C = self.model_config.pacejka_c
        self.pacejka_Df = self.model_config.pacejka_d_front
        self.pacejka_Dr = self.model_config.pacejka_d_rear
        self.mu = self.model_config.wheel_friction
        self.g = self.model_config.gravity

        # symbolic variables
        self.sym_vx = ca.SX.sym(
            'vx')  # body fram vx, vy (vx>0 points in direction car points, vy>0 points to left hand side)
        self.sym_vy = ca.SX.sym('vy')
        self.sym_psidot = ca.SX.sym('psidot')
        self.sym_epsi = ca.SX.sym('epsi')
        self.sym_s = ca.SX.sym('s')
        self.sym_xtran = ca.SX.sym('xtran')
        self.sym_u_s = ca.SX.sym('gamma')
        self.sym_u_a = ca.SX.sym('a')

        self.sym_c = self.get_curvature(self.sym_s)

        # Slip angles and Pacejka tire forces
        self.sym_alpha_f = -ca.atan2(
            (self.sym_vy + self.L_f * self.sym_psidot) * ca.cos(self.sym_u_s) - self.sym_vx * ca.sin(self.sym_u_s),
            self.sym_vx * ca.cos(self.sym_u_s) + (self.sym_vy + self.L_f * self.sym_psidot) * ca.sin(self.sym_u_s))
        self.sym_alpha_r = -ca.atan2(self.sym_vy - self.L_r * self.sym_psidot, self.sym_vx)

        self.sym_fyf = self.pacejka_Df * ca.sin(self.pacejka_C * ca.atan(self.pacejka_B * self.sym_alpha_f))
        self.sym_fyr = self.pacejka_Dr * ca.sin(self.pacejka_C * ca.atan(self.pacejka_B * self.sym_alpha_r))

        # instantaneous accelerations
        self.sym_ax = self.sym_u_a - self.c_f * self.sym_vx - self.sym_fyf * ca.sin(self.sym_u_s) / self.m
        self.sym_ay = (self.sym_fyf * ca.cos(self.sym_u_s) + self.sym_fyr) / self.m
        self.sym_wz = (self.L_f * self.sym_fyf * ca.cos(self.sym_u_s) - self.L_r * self.sym_fyr) / self.I_z

        # time derivatives
        self.sym_dvx = self.sym_ax + self.sym_psidot * self.sym_vy
        self.sym_dvy = self.sym_ay - self.sym_psidot * self.sym_vx
        self.sym_dpsidot = self.sym_wz
        self.sym_depsi = self.sym_psidot - self.sym_c * (self.sym_vx * ca.cos(self.sym_epsi) - self.sym_vy * ca.sin(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c)
        self.sym_ds = (self.sym_vx * ca.cos(self.sym_epsi) - self.sym_vy * ca.sin(self.sym_epsi)) / (1 - self.sym_xtran * self.sym_c)
        self.sym_dxtran = self.sym_vx * ca.sin(self.sym_epsi) + self.sym_vy * ca.cos(self.sym_epsi)

        # state and state derivative functions
        self.sym_q = ca.vertcat(self.sym_vx, self.sym_vy, self.sym_psidot, self.sym_epsi, self.sym_s, self.sym_xtran)
        self.sym_u = ca.vertcat(self.sym_u_a, self.sym_u_s)
        self.sym_dq = ca.vertcat(self.sym_dvx, self.sym_dvy, self.sym_dpsidot, self.sym_depsi, self.sym_ds,
                                 self.sym_dxtran)

        self.faccel = ca.Function('faccel', [self.sym_q, self.sym_u], [self.sym_ax, self.sym_ay], self.options('faccel'))

        self.precompute_model()
        return

    def state2qu(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.p.e_psi, state.p.s, state.p.x_tran])
        u = np.array([state.u.u_a, state.u.u_steer])
        return q, u

    def state2q(self, state: VehicleState) -> np.ndarray:
        q = np.array([state.v.v_long, state.v.v_tran, state.w.w_psi, state.p.e_psi, state.p.s, state.p.x_tran])
        return q

    def input2u(self, input: VehicleActuation) -> np.ndarray:
        u = np.array([input.u_a, input.u_steer])
        return u

    def q2state(self, state: VehicleState, q: np.ndarray):
        state.v.v_long = q[0]
        state.v.v_tran = q[1]
        state.w.w_psi = q[2]
        state.p.e_psi = q[3]
        state.p.s = q[4]
        state.p.x_tran = q[5]
        return

    def u2input(self, input: VehicleActuation, u: np.ndarray):
        input.u_a = u[0]
        input.u_steer = u[1]
        return

    def qu2state(self, state: VehicleState, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            state.v.v_long = q[0]
            state.v.v_tran = q[1]
            state.w.w_psi = q[2]
            state.p.e_psi = q[3]
            state.p.s = q[4]
            state.p.x_tran = q[5]
        if u is not None:
            state.u.u_a = u[0]
            state.u.u_steer = u[1]
        return

    def qu2prediction(self, prediction: VehiclePrediction, q: np.ndarray = None, u: np.ndarray = None):
        if q is not None:
            prediction.v_long = array.array('d', q[:, 0])
            prediction.v_tran = array.array('d', q[:, 1])
            prediction.psidot = array.array('d', q[:, 2])
            prediction.e_psi = array.array('d', q[:, 3])
            prediction.s = array.array('d', q[:, 4])
            prediction.x_tran = array.array('d', q[:, 5])
        if u is not None:
            prediction.u_a = array.array('d', u[:, 0])
            prediction.u_steer = array.array('d', u[:, 1])


def get_dynamics_model(t_start: float, model_config: DynamicsConfig, track=None) -> CasadiDynamicsModel:
    '''
    Helper function for getting a vehicle model class from a text string
    Should be used anywhere vehicle models may be changed by configuration
    '''
    if model_config.model_name == 'dynamic_bicycle_cl':
        return CasadiDynamicCLBicycle(t_start, model_config, track=track)
    elif model_config.model_name == 'dynamic_bicycle_full':
        return CasadiDynamicBicycleFull(t_start, model_config, track=track)
    else:
        raise ValueError('Unrecognized vehicle model name: %s' % model_config.model_name)


if __name__ == '__main__':
    pass
