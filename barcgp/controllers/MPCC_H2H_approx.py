#!/usr/bin python3
import warnings
from typing import List

import casadi
import numpy as np

import casadi as ca

import sys, os, pathlib

sys.path.append(os.path.join(os.path.expanduser('~'), 'forces_pro_client'))
import forcespro
import forcespro.nlp

from barcgp.dynamics.models.dynamics_models import CasadiDynamicsModel
from barcgp.common.pytypes import VehicleState, VehicleActuation, VehiclePrediction
from barcgp.dynamics.models.obstacle_types import RectangleObstacle

from barcgp.controllers.abstract_controller import AbstractController
from barcgp.controllers.utils.controllerTypes import MPCCApproxFullModelParams
from barcgp.common.utils.file_utils import *
from dataclasses import fields


class MPCC_H2H_approx(AbstractController):
    """
    Using MPCC Lag and contouring approximation!
    """

    def __init__(self, dynamics, track, control_params=MPCCApproxFullModelParams(), name=None, track_name=None):

        assert isinstance(dynamics, CasadiDynamicsModel)
        self.all_tracks = control_params.all_tracks
        self.dynamics = dynamics

        self.track = track
        self.track_name = track_name

        self.dt = control_params.dt
        self.n = control_params.n
        self.d = control_params.d

        self.slack = control_params.slack
        # Obstacle slack
        self.Q_cs = control_params.Q_cs
        self.l_cs = control_params.l_cs
        self.Q_cs_e = control_params.Q_cs_e
        self.l_cs_e = control_params.l_cs_e

        # Track slack and v_long max
        self.Q_ts = control_params.Q_ts
        self.Q_vmax = control_params.Q_vmax
        self.vlong_max_soft = control_params.vlong_max_soft

        self.N = control_params.N

        self.m = self.dynamics.model_config.mass
        self.lf = self.dynamics.model_config.wheel_dist_front
        self.lr = self.dynamics.model_config.wheel_dist_rear

        self.lencar = 0.36  # TODO: change
        self.widthcar = 0.2  # Ratio of car length to width

        # MPCC params
        self.control_params = control_params
        self.N = control_params.N
        self.Ts = control_params.dt
        self.Qc = control_params.Qc
        self.Ql = control_params.Ql
        self.Q_theta = control_params.Q_theta
        self.Q_xref = control_params.Q_xref
        self.R_d = control_params.R_d
        self.R_delta = control_params.R_delta

        # Input Box Constraints
        self.state_ub = control_params.state_ub
        self.state_lb = control_params.state_lb
        self.input_ub = control_params.input_ub
        self.input_lb = control_params.input_lb
        self.input_rate_ub = control_params.input_rate_ub
        self.input_rate_lb = control_params.input_rate_lb

        self.uvars = ['u_a', 'u_delta', 'v_proj']
        if self.slack:
            self.zvars = ['vx', 'vy', 'psidot', 'posx', 'posy', 'psi', 'e_psi', 's', 'x_tran', 'theta', 's_prev', 'u_a',
                          'u_delta', 'v_proj',
                          'u_a_prev', 'u_delta_prev', 'v_proj_prev', 'obs_slack', 'obs_slack_e']
        else:
            self.zvars = ['vx', 'vy', 'psidot', 'posx', 'posy', 'psi', 'e_psi', 's', 'x_tran', 'u_a', 'u_delta',
                          'u_a_prev', 'u_delta_prev']
        if self.slack:
            self.pvars = ['std_x',
                          'std_y',
                          'xref',
                          'xref_scale',
                          's0',
                          'N',
                          'x_tran_obs',
                          's_obs',
                          's_obs_prev',
                          'kp1_0', 'kp1_1', 'kp1_2', 'kp1_3', 'kp1_4', 'kp1_5',
                          'kp2_0', 'kp2_1', 'kp2_2', 'kp2_3', 'kp2_4', 'kp2_5',
                          'kp3_0', 'kp3_1', 'kp3_2', 'kp3_3', 'kp3_4', 'kp3_5',
                          'kp4_0', 'kp4_1', 'kp4_2', 'kp4_3', 'kp4_4', 'kp4_5',
                          'kp5_0', 'kp5_1', 'kp5_2', 'kp5_3', 'kp5_4', 'kp5_5',
                          'track_length',
                          'Qc', 'Ql', 'Q_theta', 'Q_xref', 'R_d', 'R_delta',
                          'r', 'x_ob', 'y_ob', 's', 'psi_ob', 'l_ob', 'w_ob', 'deactivate_ob', 'Q_cs', 'l_cs', 'Q_ts',
                          'Q_vmax', 'Q_cs_e', 'l_cs_e', 'vlong_max_soft']
        else:
            self.pvars = ['xref', 'xt', 'yt', 'psit', 'theta_hat', 'Qc', 'Ql', 'Q_theta', 'Q_xref', 'R_d', 'R_delta', \
                          'r', 'x_ob', 'y_ob', 's', 'psi_ob', 'l_ob', 'w_ob', 'deactivate_ob']

        self.half_width = track.half_width
        self.track_length = track.track_length

        self.num_std_deviations = control_params.num_std_deviations
        self.policy_map = {None: self.no_blocking_policy, "aggressive_blocking": self.aggressive_blocking_policy,
                           "only_left": self.only_left_blocking_policy, "only_right": self.only_right_blocking_policy}

        self.optlevel = control_params.optlevel

        self.solver_dir = control_params.solver_dir

        self.u_prev = np.zeros(self.d)
        self.x_pred = np.zeros((self.N, self.n))
        self.u_pred = np.zeros((self.N, self.d))
        self.x_ws = None
        self.u_ws = None

        self.model = None
        self.options = None
        self.solver = None
        if name is None:
            self.solver_name = 'MPCC_H2H_solver_forces_pro'
        else:
            self.solver_name = name
        self.state_input_prediction = VehiclePrediction()

        self.initialized = False

        self.first = True
        self.theta_prev = []
        self.s_prev = []
        self.n_sol_count = 0

    def initialize(self):
        if self.solver_dir:
            self.solver_dir = pathlib.Path(self.solver_dir).expanduser()  # allow the use of ~
            self._load_solver(self.solver_dir)
        else:
            self._build_solver()

        self.initialized = True

    def get_prediction(self):
        return self.state_input_prediction.copy()

    def set_warm_start(self, x_ws, u_ws):
        if x_ws.shape[0] != self.N or x_ws.shape[1] != self.n:  # TODO: self.N+1
            raise (RuntimeError(
                'Warm start state sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    x_ws.shape[0], x_ws.shape[1], self.N, self.n)))
        if u_ws.shape[0] != self.N or u_ws.shape[1] != self.d:
            raise (RuntimeError(
                'Warm start input sequence of shape (%i,%i) is incompatible with required shape (%i,%i)' % (
                    u_ws.shape[0], u_ws.shape[1], self.N, self.d)))

        self.x_ws = x_ws
        self.u_ws = u_ws

    def step(self, ego_state: VehicleState, tv_state: VehicleState, tv_pred: VehiclePrediction = None,
             policy: str = None, logger=None, num_lap=0):

        # evaluate policy for behavior
        policy = self.policy_map.get(policy)
        ego_state.p.s += self.track_length * num_lap
        tv_state.p.s += self.track_length * num_lap
        if abs((tv_state.p.s + self.track_length) - ego_state.p.s) < abs(tv_state.p.s - ego_state.p.s):
            tv_state.p.s += self.track_length
        elif abs((tv_state.p.s - self.track_length) - ego_state.p.s) < abs(tv_state.p.s - ego_state.p.s):
            tv_state.p.s -= self.track_length
        x_ref_blocking, blocking = policy(ego_state, tv_state)
        x_ref = np.tile(x_ref_blocking, (self.N,))
        xref_scale = max(ego_state.p.s - tv_state.p.s, 0) if tv_state is not None else 0

        # Initialize Obstacle List
        obstacle = list()
        # if tv_state is not None and tv_state.t and tv_pred is not None:
        #     # In case where we don't get TV prediction, just tile the first state for full horizon
        #     obstacle = [RectangleObstacle(xc=tv_state.x.x, yc=tv_state.x.y, psi=tv_state.e.psi, s=tv_state.p.s,
        #                                   x_tran=tv_state.p.x_tran,
        #                                   h=self.lencar, w=self.widthcar) for i in range(self.N + 1)]
        # else:
        for _ in range(self.N + 1):
            obstacle.append(RectangleObstacle())
        # if tv_state is not None and tv_state.t:
        #     obstacle[0] = RectangleObstacle(xc=tv_state.x.x, yc=tv_state.x.y, psi=tv_state.e.psi, s=tv_state.p.s,
        #                                   x_tran=tv_state.p.x_tran,
        #                                   h=self.lencar, w=self.widthcar)

        # find out if prediction is parametric, global or both

        # if not blocking, we need to use all predictions! Otherwise, only interested in the current one to avoid
        # crashing
        # if not blocking and tv_pred is not None and tv_pred.t and tv_state.p.s - ego_state.p.s < 4 *self.lencar:
        if tv_pred is not None and tv_pred.t and tv_state is not None and tv_state.t:
            # contains_parametric = tv_pred.s and np.any(tv_pred.s)
            # contains_global = tv_pred.x and np.any(tv_pred.x)
            contains_parametric = np.any(tv_pred.s)
            contains_global = np.any(tv_pred.x)
            offs = 0
            t_ = tv_pred.t
            while t_ < tv_state.t - 0.5*self.dt:
                offs += 1
                t_ += self.dt

            if contains_parametric and contains_global:
                for i, (s, x_tran, x, y, psi) in enumerate(
                        zip(tv_pred.s[offs:self.N], tv_pred.x_tran[offs:self.N], tv_pred.x[offs:self.N],
                            tv_pred.y[offs:self.N], tv_pred.psi[offs:self.N])):
                    if i > self.N:
                        break
                    xy_std = tv_pred.xy_cov[offs:][i] if tv_pred.xy_cov is not None else np.zeros((2, 2))
                    obstacle[i] = RectangleObstacle(xc=x, yc=y, psi=psi, s=s, x_tran=x_tran,
                                                        h=self.lencar, w=self.widthcar,
                                                        std_local_x=self.num_std_deviations * np.sqrt(xy_std[0, 0]),
                                                        std_local_y=self.num_std_deviations * np.sqrt(xy_std[1, 1]))
            elif contains_parametric:
                for i, (s, x_tran, e_psi) in enumerate(zip(tv_pred.s[offs:self.N], tv_pred.x_tran[offs:self.N],
                                                           tv_pred.e_psi[offs:self.N])):
                    if i > self.N:
                        break
                    xy_std = tv_pred.xy_cov[offs:][i] if tv_pred.xy_cov is not None else np.zeros((2, 2))
                    global_coord = self.track.local_to_global((s, x_tran, e_psi))
                    obstacle[i] = RectangleObstacle(xc=global_coord[0], yc=global_coord[1], psi=global_coord[2],
                                                        s=s, x_tran=x_tran,
                                                        h=self.lencar, w=self.widthcar,
                                                        std_local_x=self.num_std_deviations * np.sqrt(xy_std[0, 0]),
                                                        std_local_y=self.num_std_deviations * np.sqrt(xy_std[1, 1]))
            elif contains_global:
                # TODO Add local state
                for i, (x, y, psi) in enumerate(
                        zip(tv_pred.x[offs:self.N], tv_pred.y[offs:self.N], tv_pred.psi[offs:self.N])):
                    if i > self.N:
                        break
                    xy_std = tv_pred.xy_cov[offs:][i] if tv_pred.xy_cov is not None else np.zeros((2, 2))
                    obstacle[i] = RectangleObstacle(xc=x, yc=y, psi=psi, h=self.lencar, w=self.widthcar,
                                                        std_local_x=self.num_std_deviations * np.sqrt(xy_std[0, 0]),
                                                        std_local_y=self.num_std_deviations * np.sqrt(xy_std[1, 1]))

        control, info, exitflag = self.solve(ego_state, ego_state.p.s, x_ref, xref_scale, obstacle, blocking)

        ego_state.u.u_a = control.u_a
        ego_state.p.s -= self.track_length*num_lap
        ego_state.u.u_steer = control.u_steer
        if logger is not None:
            x_vals = []
            for obs in obstacle:
                x_vals.append(obs.xc)
            logger(f"X current: {tv_state.x.x}")
            logger(f"X obs: {x_vals[0]}")
        return info, blocking, exitflag

    def solve(self, state: VehicleState, s0, x_ref: np.array, xref_scale, obstacle: List[RectangleObstacle], blocking):
        if not self.initialized:
            raise (RuntimeError(
                'MPCC controller is not initialized, run MPCC.initialize() before calling MPCC.solve()'))

        x, _ = self.dynamics.state2qu(state)

        if self.x_ws is None:
            warnings.warn('Initial guess of open loop state sequence not provided, using zeros')
            self.x_ws = np.zeros((self.N, self.n))
        if self.u_ws is None:
            warnings.warn('Initial guess of open loop input sequence not provided, using zeros')
            self.u_ws = np.zeros((self.N, self.d))

        # Set up real-time parameters, warm-start, and key_pts declaration
        parameters = []
        initial_guess = []
        key_pts = []
        current_s = state.p.s
        while current_s < 0: current_s += self.track.track_length
        while current_s >= self.track.track_length: current_s -= self.track.track_length
        if len(self.track.key_pts) < 5:
            for i in range(len(self.track.key_pts)):
                key_pts.append(self.track.key_pts[i])
            while len(key_pts) < 5:
                key_pts.append(key_pts[-1])
        else:
            key_pt_idx_s = np.where(current_s >= self.track.key_pts[:, 3])[0][-1] - 1
            if key_pt_idx_s == -1:
                key_pt_idx_s = len(self.track.key_pts) - 1
            difference = max(0, (key_pt_idx_s + 4) - (len(self.track.key_pts) - 1))
            difference_ = difference
            while difference > 0:
                key_pts.append(self.track.key_pts[difference_ - difference])
                difference -= 1
            for i in range(5 - len(key_pts)):
                key_pts.append(self.track.key_pts[key_pt_idx_s + i])

        for stageidx in range(self.N):
            # Default to respecting obstacles
            obs_deactivate = False
            # deactivate prediction obstacle avoidance if in blocking-mode
            if (blocking) and (stageidx > 0 or obstacle[0] is None):
                obs_deactivate = True
            if obstacle[stageidx].h == 0:
                obs_deactivate = True

            initial_guess.append(self.x_ws[stageidx])  # x
            initial_guess.append(self.u_ws[stageidx])  # u
            initial_guess.append(self.u_ws[stageidx - 1])  # u_prev
            stage_p = []
            stage_p.append(obstacle[stageidx].std_local_x)
            stage_p.append(obstacle[stageidx].std_local_y)
            stage_p.append(x_ref[stageidx])  # x_ref (for blocking maneuvers)
            stage_p.append(xref_scale)  # x_ref (for blocking maneuvers)
            stage_p.append(s0 if stageidx == self.N - 1 else -100)
            stage_p.append(stageidx)
            stage_p.append(obstacle[stageidx].x_tran if not blocking else -100)
            stage_p.append(obstacle[stageidx].s if not blocking else -100)
            stage_p.append((obstacle[stageidx - 1].s if stageidx > 0 else np.inf) if not blocking else -100)
            stage_p.extend(key_pts[0])
            stage_p.extend(key_pts[1])
            stage_p.extend(key_pts[2])
            stage_p.extend(key_pts[3])
            stage_p.extend(key_pts[4])
            stage_p.extend([self.track.track_length])
            stage_p.extend([self.Qc, self.Ql, self.Q_theta, self.Q_xref, self.R_d, self.R_delta, self.half_width])
            stage_p.extend([
                obstacle[stageidx].xc,
                obstacle[stageidx].yc,
                obstacle[stageidx].s,
                obstacle[stageidx].psi,
                obstacle[stageidx].h,
                obstacle[stageidx].w,
                obs_deactivate  # deactivate obstacle by default
            ])

            parameters.append(stage_p)

            # 2 slack constraints, 4 cost variables
            if self.slack:
                initial_guess.append(np.zeros((2,)))
                parameters.extend(np.array([self.Q_cs / ((1 + 2 * stageidx)), self.l_cs / ((1 + 2 * stageidx)),
                                            self.Q_ts, self.Q_vmax, self.Q_cs_e, self.l_cs_e,
                                            self.vlong_max_soft]).reshape((7, -1)))

        parameters = np.concatenate(parameters)
        initial_guess = np.concatenate(initial_guess)

        # problem dictionary, arrays have to be flattened
        problem = dict()
        problem["xinit"] = np.concatenate((np.append(np.append(x, state.p.s), state.p.s), self.u_prev))
        problem["all_parameters"] = parameters
        problem["x0"] = initial_guess

        output, exitflag, solve_info = self.solver.solve(problem)

        if exitflag == 1:
            if exitflag == 0:
                info = {"success": False, "return_status": "Successfully Solved", "solve_time": solve_info.solvetime,
                        "info": solve_info}
            else:
                info = {"success": True, "return_status": "Successfully Solved", "solve_time": solve_info.solvetime,
                        "info": solve_info}

            for k in range(self.N):
                sol = output["x%02d" % (k + 1)]
                self.x_pred[k, :] = sol[:self.n]
                self.u_pred[k, :] = sol[self.n:self.n + self.d]

            # Construct initial guess for next iteration
            x_ws = self.x_pred[1:]
            u_ws = self.u_pred[1:]
            if self.all_tracks:
                q_integrated = np.array(self.dynamics.f_d_rk4_forces(x_ws[-1, :-2], self.track.track_length, ca.transpose(
                        ca.horzcat(key_pts[0], key_pts[1], key_pts[2], key_pts[3], key_pts[4])),
                                                                 u_ws[-1, :-1]))
                x_ws = np.vstack((x_ws,
                                  np.append(q_integrated,
                                            (q_integrated[7][0], x_ws[-1, -1]))))  # TODO fix this
            else:
                q_integrated = np.array(self.dynamics.f_d_rk4(x_ws[-1, :-2],
                                                          u_ws[-1, :-1]))
                x_ws = np.vstack((x_ws,
                                  np.append(q_integrated,
                                            (q_integrated[7][0], x_ws[-1, -1]))))  # TODO fix this
            u_ws = np.vstack((u_ws, u_ws[-1]))  # stack previous input
            self.set_warm_start(x_ws, u_ws)

            u = self.u_pred[0]
            self.n_sol_count = 0

            self.dynamics.qu2prediction(self.state_input_prediction, self.x_pred, self.u_pred)


        else:
            info = {"success": False, "return_status": 'Solving Failed, exitflag = %d' % exitflag, "solve_time": None,
                    "info": solve_info}
            self.n_sol_count += 1
            u = self.u_pred[self.n_sol_count] if self.n_sol_count < self.u_pred.shape[0] else np.zeros((self.d))

        self.u_prev = u

        control = VehicleActuation()
        self.dynamics.u2input(control, u)

        return control, info, exitflag

    def _load_solver(self, solver_dir):
        solver_found = True

        params_path = os.path.join(solver_dir, 'params.pkl')
        if not dir_exists(params_path):
            print("====================")
            print("Config File does not exist, rebuilding and saving config file to", params_path, "...")
            print("====================")
            solver_found = False
        else:
            dbfile = open(params_path, 'rb')
            p = pickle.load(dbfile)
            params = p[0]
            t_name = p[1]
            dbfile.close()
            if solver_found:
                for field in fields(params):
                    if not field.name == "solver_dir":
                        if hasattr(getattr(params, field.name), '__len__'):
                            if not np.array_equal(getattr(params, field.name),
                                                  getattr(self.control_params, field.name)):
                                solver_found = False
                                print("====================")
                                print(
                                    "Config File does not match the config this solver was built for, difference found in "
                                    "field", field.name, "and possibly others!")
                                print("Rebuilding ...")
                                print("====================")
                                break
                        else:
                            if not (getattr(params, field.name) == getattr(self.control_params, field.name)):
                                solver_found = False
                                print("====================")
                                print(
                                    "Config File does not match the config this solver was built for, difference found in field",
                                    field.name, ":", getattr(params, field.name), "/",
                                    getattr(self.control_params, field.name), "(current/config), and possibly others!")
                                print("Rebuilding ...")
                                print("====================")
                                break
            if solver_found and self.all_tracks and self.track_name == 'track':
                print('Controller built for short scenario tracks, usually performs poorly on full track, consider '
                      'rebuilding!')
            elif solver_found and (not self.all_tracks) and (not t_name == self.track_name):
                if self.track_name is None:
                    print('Controller built ONLY for track "', t_name,
                          '" while current track is unknown (None). Consider rebuilding if not intended!')
                else:
                    solver_found = False
                    print("====================")
                    print('Controller built ONLY for track "', t_name, '" which differs from "', self.track_name, '"')
                    print('Rebuilding ...')
                    print("====================")
            elif solver_found and (not self.all_tracks) and (not t_name == "track"):
                print('Controller build ONLY for one instance of track type "', t_name,
                      '", rebuild if this is not intended!')

        if solver_found:
            print('Found matching config file for solver', self.solver_name)
        else:
            self._build_solver()

        self.solver = forcespro.nlp.Solver.from_directory(solver_dir)

    def _build_solver(self):
        def nonlinear_ineq(z, p):
            # Extract states
            posx = z[self.zvars.index('posx')]
            posy = z[self.zvars.index('posy')]
            psi = z[self.zvars.index('psi')]

            if self.slack:
                obstacle_slack = z[self.zvars.index('obs_slack')]
                obstacle_slack_e = z[self.zvars.index('obs_slack_e')]

            # Ellipsoidal obstacle
            std_x = p[self.pvars.index('std_x')]
            std_y = p[self.pvars.index('std_y')]
            x_ob = p[self.pvars.index('x_ob')]
            y_ob = p[self.pvars.index('y_ob')]
            psi_ob = p[self.pvars.index('psi_ob')]
            l_ob = p[self.pvars.index('l_ob')]
            w_ob = p[self.pvars.index('w_ob')]
            deactivate_ob = p[self.pvars.index('deactivate_ob')]
            step = p[self.pvars.index('N')]

            s = ca.sin(psi_ob)
            c = ca.cos(psi_ob)

            # 4 Disks to approximate box car shape
            r_d = self.widthcar / 1.8
            dl = self.lencar * 0.9 / 3
            s_e = ca.sin(psi)
            c_e = ca.cos(psi)

            dx1 = x_ob - (posx - 3 * dl * c_e / 2)
            dx2 = x_ob - (posx - dl * c_e / 2)
            dx3 = x_ob - (posx + dl * c_e / 2)
            dx4 = x_ob - (posx + 3 * dl * c_e / 2)
            dy1 = y_ob - (posy - 3 * dl * s_e / 2)
            dy2 = y_ob - (posy - dl * s_e / 2)
            dy3 = y_ob - (posy + dl * s_e / 2)
            dy4 = y_ob - (posy + 3 * dl * s_e / 2)

            if self.slack:
                # this is the minimum allowed distance in TV heading direction.
                # It is scaled up by std_x, std_y (uncertainty along TV heading)
                # Within this scaling, violation of the minimum distance is allowed due to slack
                # hard constraint on collision!
                slack_l_hard = ca.if_else(step > -1, 0.2 * (self.lencar - self.lencar * obstacle_slack_e / 10), 0)
                slack_w_hard = ca.if_else(step > -1, 0.2 * (self.widthcar - self.widthcar * obstacle_slack_e / 10), 0)
                slack_l = ca.if_else(std_x == 0, slack_l_hard,
                                     + std_x * 1 - std_x * obstacle_slack + slack_l_hard)
                slack_w = ca.if_else(std_y == 0, slack_w_hard,
                                     + std_y * 1 - std_y * obstacle_slack + slack_w_hard)

                a = (l_ob / np.sqrt(2) + r_d + slack_l)
                b = (w_ob / np.sqrt(2) + r_d + slack_w)

            # deactivate_ob = ca.if_else(ca.logic_and(s_prev > s_obs_prev + 0.02, step>1), 1, deactivate_ob)
            i1 = (c * dx1 - s * dy1) ** 2 * 1 / (a ** 2 + 0.0001) + (s * dx1 + c * dy1) ** 2 * 1 / (b ** 2 + 0.001)
            i2 = (c * dx2 - s * dy2) ** 2 * 1 / (a ** 2 + 0.0001) + (s * dx2 + c * dy2) ** 2 * 1 / (b ** 2 + 0.001)
            i3 = (c * dx3 - s * dy3) ** 2 * 1 / (a ** 2 + 0.0001) + (s * dx3 + c * dy3) ** 2 * 1 / (b ** 2 + 0.001)
            i4 = (c * dx4 - s * dy4) ** 2 * 1 / (a ** 2 + 0.0001) + (s * dx4 + c * dy4) ** 2 * 1 / (b ** 2 + 0.001)

            obsval1 = casadi.if_else(deactivate_ob == 1, 2, i1)
            obsval2 = casadi.if_else(deactivate_ob == 1, 2, i2)
            obsval3 = casadi.if_else(deactivate_ob == 1, 2, i3)
            obsval4 = casadi.if_else(deactivate_ob == 1, 2, i4)

            # Concatenate
            hval = ca.vertcat(-1, 1, obsval1, obsval2, obsval3, obsval4)
            return hval

        def objective_cost(z, p):
            # Track params (Key points, length and width)
            kp1 = p[self.pvars.index('kp1_0'):self.pvars.index('kp1_0') + 6]
            kp2 = p[self.pvars.index('kp2_0'):self.pvars.index('kp2_0') + 6]
            kp3 = p[self.pvars.index('kp3_0'):self.pvars.index('kp3_0') + 6]
            kp4 = p[self.pvars.index('kp4_0'):self.pvars.index('kp4_0') + 6]
            kp5 = p[self.pvars.index('kp5_0'):self.pvars.index('kp5_0') + 6]
            t_l = p[self.pvars.index('track_length')]
            r = p[self.pvars.index('r')]

            # Cost and behavior params
            Qc = p[self.pvars.index('Qc')]
            Ql = p[self.pvars.index('Ql')]
            Q_theta = p[self.pvars.index('Q_theta')]
            R_d = p[self.pvars.index('R_d')]
            R_delta = p[self.pvars.index('R_delta')]
            Q_xref = p[self.pvars.index('Q_xref')]
            vlong_max_soft = p[self.pvars.index('vlong_max_soft')]

            # Blocking reference
            xref = p[self.pvars.index('xref')]
            xref_scale = p[self.pvars.index('xref_scale')]

            # States
            v_long = z[self.zvars.index('vx')]
            posx = z[self.zvars.index('posx')]
            posy = z[self.zvars.index('posy')]
            x_tran = z[self.zvars.index('x_tran')]  # todo: replace
            deactivate_ob = p[self.pvars.index('deactivate_ob')]

            # progress states
            theta = z[self.zvars.index('theta')]
            v_proj_prev = z[self.zvars.index('v_proj_prev')]

            # Obstacle avoidance params
            std_x = p[self.pvars.index('std_x')]
            std_y = p[self.pvars.index('std_y')]
            Q_xref /= ca.power((1 + xref_scale), 2)

            if self.slack:
                obstacle_slack = z[self.zvars.index('obs_slack')]
                Q_cs = p[self.pvars.index('Q_cs')]
                l_cs = p[self.pvars.index('l_cs')]

                Q_ts = p[self.pvars.index('Q_ts')]
                Q_vmax = p[self.pvars.index('Q_vmax')]

                obstacle_slack_e = z[self.zvars.index('obs_slack_e')]
                Q_cs_e = p[self.pvars.index('Q_cs_e')]
                l_cs_e = p[self.pvars.index('l_cs_e')]

            # Extract inputs
            u_a = z[self.zvars.index('u_a')]
            u_delta = z[self.zvars.index('u_delta')]
            u_a_prev = z[self.zvars.index('u_a_prev')]
            u_delta_prev = z[self.zvars.index('u_delta_prev')]
            u_a_dot = u_a - u_a_prev
            u_delta_dot = u_delta - u_delta_prev

            # Approximate linearized contouring and lag error
            xt_hat, yt_hat, psit = self.track.local_to_global_ca(theta, 0, 0, ca.transpose(
                ca.horzcat(kp1, kp2, kp3, kp4, kp5)), t_l, self.all_tracks)
            sin_psit = ca.sin(psit)
            cos_psit = ca.cos(psit)
            e_cont = -sin_psit * (xt_hat - posx) + cos_psit * (yt_hat - posy)
            e_lag = cos_psit * (xt_hat - posx) + sin_psit * (yt_hat - posy)

            # if self.solver_name == 'mpcc_h2h_tv_lab_track':
            #     e_cont -= 0.2

            # Contouring + Lag + Input - Progress + delta input
            cost = ca.bilin(Qc, e_cont, e_cont) + ca.bilin(Ql, e_lag, e_lag) + \
                   ca.bilin(R_d, u_a, u_a) + ca.bilin(R_delta, u_delta, u_delta) - Q_theta * v_proj_prev * self.dt + \
                   ca.bilin(R_d, u_a_dot, u_a_dot) + ca.bilin(R_delta, u_delta_dot, u_delta_dot)

            # Reference tracking cost todo: replace xtran by e_cont and test
            cost += ca.if_else(xref == -20, 0, ca.bilin(Q_xref, -e_cont - xref, -e_cont - xref))

            if self.slack:
                # Obstacle Slack, scale by size
                avg_std = (std_x + std_y) / 2
                cost += ca.if_else(ca.logic_or(deactivate_ob, avg_std == 0), 0,
                                   (ca.power(obstacle_slack, 2) * Q_cs + obstacle_slack * l_cs) / avg_std)
                # Inner Safety bound, extreme slack
                cost += ca.if_else(deactivate_ob, 0, ca.power(obstacle_slack_e, 2) * Q_cs_e + obstacle_slack_e * l_cs_e)
                # Reference Speeds todo: only punish > v_max, not < v_max!
                cost += ca.fmax(v_long - vlong_max_soft, 0) ** 2 * Q_vmax
                # Track boundary constraints
                cost += ca.fmax(ca.fabs(e_cont) - (r - self.widthcar), 0) ** 2 * Q_ts
            return cost

        def stage_input_rate_constraint(z, p):
            # Extract inputs
            u_a = z[self.zvars.index('u_a')]
            u_delta = z[self.zvars.index('u_delta')]
            v_proj = z[self.zvars.index('v_proj')]

            u_a_prev = z[self.zvars.index('u_a_prev')]
            u_delta_prev = z[self.zvars.index('u_delta_prev')]
            v_proj_prev = z[self.zvars.index('v_proj_prev')]

            u_a_dot = u_a - u_a_prev
            u_delta_dot = u_delta - u_delta_prev
            v_proj_dot = v_proj - v_proj_prev

            return ca.vertcat(u_a_dot, u_delta_dot, v_proj_dot)

        def stage_equality_constraint(z, p):
            q = z[:self.n - 2]
            # Get u states, shifted by two due to extra approximation states
            u = z[self.n:self.n + self.d]
            kp1 = p[self.pvars.index('kp1_0'):self.pvars.index('kp1_0') + 6]
            kp2 = p[self.pvars.index('kp2_0'):self.pvars.index('kp2_0') + 6]
            kp3 = p[self.pvars.index('kp3_0'):self.pvars.index('kp3_0') + 6]
            kp4 = p[self.pvars.index('kp4_0'):self.pvars.index('kp4_0') + 6]
            kp5 = p[self.pvars.index('kp5_0'):self.pvars.index('kp5_0') + 6]
            t_l = p[self.pvars.index('track_length')]
            if self.all_tracks:
                integrated = self.dynamics.f_d_rk4_forces(q, t_l, ca.transpose(ca.horzcat(kp1, kp2, kp3, kp4, kp5)),
                                                          u[:-1])
            else:
                integrated = self.dynamics.f_d_rk4(q, u[:-1])
            # theta next is theta prev + vk/Ts
            return ca.vertcat(integrated, z[self.zvars.index('theta')] + z[self.zvars.index('v_proj')] * self.dt,
                              q[self.zvars.index('s')], u)

        # Forces model
        self.model = forcespro.nlp.SymbolicModel(self.N)

        # Number of parameters
        self.model.nvar = self.n + self.d + self.d  # Stage variables z = [x, theta, u, u_prev]

        if self.slack:
            self.model.nvar += 1 + 1  # Stage variables z = [x, u, u_prev, 1 slacks, 1 extreme slack]

        self.model.nh = self.d + 6  # of inequality constraints

        self.model.neq = self.n + self.d  # of equality constraints
        self.model.npar = len(self.pvars)  # number of real-time parameters (p)

        self.model.objective = lambda z, p: objective_cost(z, p)

        # dynamics only in state Variables
        stage_E_q = np.hstack((np.eye(self.n), np.zeros((self.n, self.d + self.d))))
        stage_E_u = np.hstack((np.zeros((self.d, self.n + self.d)), np.eye(self.d)))
        stage_E = np.vstack((stage_E_q, stage_E_u))

        # Equality constraints
        if self.slack:
            stage_E = np.hstack((stage_E, np.zeros((stage_E.shape[0], 1))))
            stage_E = np.hstack((stage_E, np.zeros((stage_E.shape[0], 1))))
            # stage_E = np.hstack((stage_E, np.zeros((stage_E.shape[0], 1))))  # TODO

        self.model.E = stage_E
        self.model.eq = stage_equality_constraint

        # Inequality constraint bounds
        self.model.ineq = lambda z, p: ca.vertcat(nonlinear_ineq(z, p),  # ego, enemy states
                                                  stage_input_rate_constraint(z, p))

        # Nonlinear constraint bounds
        # upper_track_bound, lower_track_bound, obsval1, obsval2, obsval3, obsval4

        self.model.hu = np.concatenate((np.array([0.0, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf]),
                                        self.input_rate_ub * self.dt))  # upper bound for nonlinear constraints, pg. 23 of FP manual
        self.model.hl = np.concatenate((np.array([-ca.inf, 0.0, 1.0, 1.0, 1.0, 1.0]), self.input_rate_lb * self.dt))

        # Input box constraints
        if self.slack:
            self.model.ub = np.concatenate(
                (self.state_ub, [self.state_ub[self.zvars.index('s')]], self.input_ub, self.input_ub, np.array([1]), np.array([
                    15])))  # [x, u, u_prev, obs_slack, track_slack, obs_slack_e] track slack is constrained t 100 to avoid INF
            self.model.lb = np.concatenate(
                (self.state_lb, [self.state_lb[self.zvars.index('s')]], self.input_lb, self.input_lb,
                 np.array([0]), np.array([0])))  # [x, u, u_prev, obs_slack, track_slack]
        else:
            self.model.ub = np.concatenate(
                (self.state_ub, [self.state_ub[self.zvars.index('s')]], self.input_ub, self.input_ub))  # [x, u, u_prev]
            self.model.lb = np.concatenate(
                (self.state_lb, [self.state_lb[self.zvars.index('s')]], self.input_lb, self.input_lb))  # [x, u, u_prev]

        # Put initial condition on all state variables x
        self.model.xinitidx = list(np.arange(self.n)) + list(
            np.arange(self.n + self.d, self.n + self.d + self.d))  # x, u_prev
        # Set solver options
        self.options = forcespro.CodeOptions(self.solver_name)
        self.options.maxit = 400
        self.options.overwrite = True
        self.options.printlevel = 0
        self.options.optlevel = self.optlevel
        self.options.BuildSimulinkBlock = False
        self.options.cleanup = True
        self.options.platform = 'Generic'
        self.options.gnu = True
        self.options.sse = True
        self.options.noVariableElimination = False
        # self.options.parallel = 1

        self.options.nlp.linear_solver = 'symm_indefinite'
        self.options.nlp.stack_parambounds = True
        self.options.nlp.TolStat = 1e-4
        self.options.nlp.TolEq = 1e-4
        self.options.nlp.TolIneq = 1e-4
        self.options.nlp.TolComp = 1e-4
        self.options.accuracy.ineq = 1e-4
        self.options.accuracy.eq = 1e-4
        self.options.accuracy.mu = 1e-4
        self.options.accuracy.rdgap = 1e-4
        self.options.solvemethod = 'PDIP_NLP'

        # Creates code for symbolic model formulation given above, then contacts server to generate new solver
        self.model.generate_solver(self.options)
        self.install_dir = self.install()  # install the model to ~/.mpclab_controllers
        # Copy solver config.
        pickle_write([self.control_params, self.track_name], os.path.join(self.install_dir, 'params.pkl'))
        self.solver = forcespro.nlp.Solver.from_directory(self.install_dir)

    def aggressive_blocking_policy(self, ego_state: VehicleState, tv_state: VehicleState,
                                   tv_prediction: VehiclePrediction = None):
        """
        Aggressive Blocking Policy. Will try to match x_tran of tv_state at all costs.
        """
        if tv_state is not None and tv_state.p.s < ego_state.p.s:
            blocking = True
            xt = tv_state.p.x_tran
            x_ref = np.sign(xt) * min(self.track.half_width, abs(float(xt)))
        else:
            # non-blocking mode
            x_ref = -20
            blocking = False
        return x_ref, blocking

    def only_right_blocking_policy(self, ego_state: VehicleState, tv_state: VehicleState,
                                   tv_prediction: VehiclePrediction = None):
        """
        Aggressive Blocking Policy. Will try to match x_tran of tv_state at all costs if TV is on the right
        """
        if tv_state is not None and tv_state.p.s < ego_state.p.s:
            blocking = True
            if tv_state.p.x_tran < ego_state.p.x_tran:
                xt = tv_state.p.x_tran
                x_ref = np.sign(xt) * min(self.track.half_width, abs(float(xt)))
            else:
                x_ref = -20
        else:
            # non-blocking mode
            x_ref = -20
            blocking = False
        return x_ref, blocking

    def only_left_blocking_policy(self, ego_state: VehicleState, tv_state: VehicleState,
                                  tv_prediction: VehiclePrediction = None):
        """
        Aggressive Blocking Policy. Will try to match x_tran of tv_state at all costs if TV is on the right
        """
        if tv_state is not None and tv_state.p.s < ego_state.p.s:
            blocking = True
            if tv_state.p.x_tran > ego_state.p.x_tran:
                xt = tv_state.p.x_tran
                x_ref = np.sign(xt) * min(self.track.half_width, abs(float(xt)))
            else:
                x_ref = -20
        else:
            # non-blocking mode
            x_ref = -20
            blocking = False
        return x_ref, blocking

    def no_blocking_policy(self, ego_state: VehicleState, tv_state: VehicleState,
                           tv_prediction: VehiclePrediction = None):
        """
        Aggressive Blocking Policy. Will try to match x_tran of tv_state at all costs if TV is on the right
        """
        if tv_state is not None and ego_state.p.s > tv_state.p.s+0.1:
            return -20, True
        return -20, False

    def get_reference_d_ca(self, s, s_ob, x_tran_ob, track_width, left=False):
        dmax = (left * track_width - left * x_tran_ob) / 2
        length = 1 * self.lencar
        dslope1 = dmax / (length) * (s - (s_ob - length))
        dslope2 = -dmax / (length) * (s - (s_ob + length)) + dmax
        return ca.if_else(ca.logic_or(s < s_ob - length, s > s_ob + length), 0,
                          ca.if_else(ca.logic_and(s > s_ob - length, s < s_ob + length), dmax,
                                     ca.if_else(s < s_ob, dslope1, dslope2)))

    def get_corridor(self, s, s_ob, x_tran_ob, track_width, left=True):
        length = 1 * self.lencar
        track_min = - track_width
        track_max = track_width
        dslope1 = ca.if_else(left, (x_tran_ob - track_min) / (length) * (s - (s_ob - length)) + track_min,
                             (x_tran_ob - track_max) / (length) * (s - (s_ob - length)) + track_max)
        dslope2 = ca.if_else(left, (track_min - x_tran_ob) / (length) * (s - (s_ob + length)) + x_tran_ob,
                             (track_max - x_tran_ob) / (length) * (s - (s_ob + length)) + x_tran_ob)
        tran_min = ca.if_else(ca.logic_or(s < s_ob - length, s > s_ob + length), track_min, ca.if_else(left, ca.if_else(
            ca.logic_and(s > s_ob - length, s < s_ob + length), x_tran_ob, ca.if_else(s < s_ob, dslope1, dslope2)),
                                                                                                       track_min))
        tran_max = ca.if_else(ca.logic_or(s < s_ob - length, s > s_ob + length), track_max, ca.if_else(left, track_max,
                                                                                                       ca.if_else(
                                                                                                           ca.logic_and(
                                                                                                               s > s_ob - length,
                                                                                                               s < s_ob + length),
                                                                                                           x_tran_ob,
                                                                                                           ca.if_else(
                                                                                                               s < s_ob,
                                                                                                               dslope1,
                                                                                                               dslope2))))

        return tran_min, tran_max
