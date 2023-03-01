#!/usr/bin/env python3
import copy
import os
import pickle
from collections import deque

import pdb

import numpy as np
import casadi as ca

from barcgp.common.utils.file_utils import *
from barcgp.common.pytypes import *
from barcgp.common.tracks.track import get_track
from barcgp.common.utils.scenario_utils import SimData, smoothPlotResults, ScenarioGenParams, ScenarioGenerator, EvalData, post_gp
from barcgp.common_control import run_pid_warmstart
from barcgp.h2h_configs import dt, N, ego_dynamics_config, ego_sim_dynamics_config, tar_dynamics_config, tar_sim_dynamics_config
from barcgp.h2h_configs import ego_mpc_params, tar_mpc_params
# from barcgp.h2h_configs import egoMax, egoMin, tarMax, tarMin

# from barcgp.dynamics.models.model_types import DynamicBicycleConfig
from barcgp.dynamics.models.dynamics_models import CasadiDynamicBicycle

from barcgp.controllers.CA_MPCC_conv import CA_MPCC_conv
# from barcgp.controllers.utils.controllerTypes import CAMPCCParams

from barcgp.simulation.dynamics_simulator import DynamicsSimulator



def main(args=None):
    ##############################################################################################################################################
    T = 20  # Max number of seconds to run experiment
    t = 0  # Initial time increment
    ##############################################################################################################################################
    factor = 3.0
    ep = 10*np.pi/180
    halfwidth = 0.55
    tar_min = VehicleState(t=0.0,
                        p=ParametricPose(s=32, x_tran=-.9 * halfwidth, e_psi=-ep),
                        v=BodyLinearVelocity(v_long=0.5*factor))
    tar_max = VehicleState(t=0.0,
                        p=ParametricPose(s=33, x_tran=.9 * halfwidth, e_psi=ep),
                        v=BodyLinearVelocity(v_long=1.0*factor))
    ego_min = VehicleState(t=0.0,
                        p=ParametricPose(s=30, x_tran=-.9 * halfwidth, e_psi=-ep),
                        v=BodyLinearVelocity(v_long=0.5*factor))
    ego_max = VehicleState(t=0.0,
                        p=ParametricPose(s=31, x_tran=.9 * halfwidth, e_psi=ep),
                        v=BodyLinearVelocity(v_long=1.0*factor))

    scen_params = ScenarioGenParams(types=['track'], egoMin=ego_min, egoMax=ego_max, tarMin=tar_min, tarMax=tar_max, width=halfwidth*2)
    scen_gen = ScenarioGenerator(scen_params)
    scenario = scen_gen.genScenario()

    track_name = scenario.track_type
    track_obj = scenario.track
    print(track_obj.track_length)

    ego_dynamics_model = CasadiDynamicBicycle(t, ego_dynamics_config, track=track_obj)
    ego_dynamics_simulator = DynamicsSimulator(t, ego_sim_dynamics_config, 
                                            track=track_obj)


    tar_dynamics_model = CasadiDynamicBicycle(t, tar_dynamics_config, track=track_obj)
    tar_dynamics_simulator = DynamicsSimulator(t, tar_sim_dynamics_config, 
                                            track=track_obj)
    
    ego_state_input_ub=VehicleState(x=Position(x=track_obj.track_extents['x_max']+1, y=track_obj.track_extents['y_max']+1),
                                e=OrientationEuler(psi=100),
                                v=BodyLinearVelocity(v_long=3.0, v_tran=3.0),
                                w=BodyAngularVelocity(w_psi=100),
                                u=VehicleActuation(u_a=2.0, u_steer=0.436))
    ego_state_input_lb=VehicleState(x=Position(x=track_obj.track_extents['x_min']-1, y=track_obj.track_extents['y_min']-1),
                                e=OrientationEuler(psi=-100),
                                v=BodyLinearVelocity(v_long=-3.0, v_tran=-2.0),
                                w=BodyAngularVelocity(w_psi=-100),
                                u=VehicleActuation(u_a=-2.0, u_steer=-0.436))
    ego_input_rate_ub = VehicleState(u=VehicleActuation(u_a=10.0, u_steer=4.5))
    ego_input_rate_lb = VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-4.5))

    tar_state_input_ub=VehicleState(x=Position(x=track_obj.track_extents['x_max']+1, y=track_obj.track_extents['y_max']+1),
                                e=OrientationEuler(psi=100),
                                v=BodyLinearVelocity(v_long=2.5, v_tran=3.0),
                                w=BodyAngularVelocity(w_psi=100),
                                u=VehicleActuation(u_a=2.0, u_steer=0.436))
    tar_state_input_lb=VehicleState(x=Position(x=track_obj.track_extents['x_min']-1, y=track_obj.track_extents['y_min']-1),
                                e=OrientationEuler(psi=-100),
                                v=BodyLinearVelocity(v_long=-3.0, v_tran=-2.0),
                                w=BodyAngularVelocity(w_psi=-100),
                                u=VehicleActuation(u_a=-2.0, u_steer=-0.436))
    tar_input_rate_ub = VehicleState(u=VehicleActuation(u_a=10.0, u_steer=4.5))
    tar_input_rate_lb = VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-4.5))

    sym_q = ca.MX.sym('q', ego_dynamics_model.n_q)
    sym_u = ca.MX.sym('u', ego_dynamics_model.n_u)
    sym_du = ca.MX.sym('du', ego_dynamics_model.n_u)

    vy_idx = 1
    wz_idx = 2
    x_idx = 3
    y_idx = 4
    psi_idx = 5

    ua_idx = 0
    us_idx = 1

    ego_sym_state_stage = ego_sym_state_term = 0
    ego_sym_input_stage = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
    ego_sym_input_term = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
    ego_sym_rate_stage = 0.5*(0.01*(sym_du[ua_idx])**2 + 0.01*(sym_du[us_idx])**2)

    ego_sym_costs = {'state': [None for _ in range(N+1)], 'input': [None for _ in range(N+1)], 'rate': [None for _ in range(N)]}
    for k in range(N):
        ego_sym_costs['state'][k] = ca.Function(f'state_stage_{k}', [sym_q], [ego_sym_state_stage])
        ego_sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [ego_sym_input_stage])
        ego_sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [ego_sym_rate_stage])
        ego_sym_costs['state'][N] = ca.Function('state_term', [sym_q], [ego_sym_state_term])
    ego_sym_costs['input'][N] = ca.Function('input_term', [sym_u], [ego_sym_input_term])

    sym_p = sym_q[[x_idx, y_idx]]
    sym_p_obs = ca.MX.sym('p_obs', 2)

    ego_radius = 0.23
    soft_ego_radius = 0.3
    tar_radius = 0.21
    obs_avoid = (ego_radius+tar_radius)**2 - ca.bilin(ca.DM.eye(2), sym_p-sym_p_obs, sym_p-sym_p_obs)
    soft_obs_avoid = (soft_ego_radius+tar_radius)**2 - ca.bilin(ca.DM.eye(2), sym_p-sym_p_obs, sym_p-sym_p_obs)
    
    f_obs_avoid = ca.Function('obs_avoid', [sym_q, sym_u, sym_p_obs], [ca.vertcat(obs_avoid, soft_obs_avoid)])
    ego_sym_constrs = {'state_input': [None] + [f_obs_avoid for _ in range(1, N+1)], 
                'rate': [None for _ in range(N)]}
    
    ego_mpc_params.soft_constraint_idxs = [[]] + [[0, 1] for _ in range(N)]
    ego_mpc_params.soft_constraint_quad = [[]] + [[100, 40] for _ in range(N)]
    ego_mpc_params.soft_constraint_lin = [[]] + [[0, 0] for _ in range(N)]

    tar_sym_state_stage = tar_sym_state_term = 0
    tar_sym_input_stage = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
    tar_sym_input_term = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
    tar_sym_rate_stage = 0.5*(0.01*(sym_du[ua_idx])**2 + 0.01*(sym_du[us_idx])**2)

    tar_sym_costs = {'state': [None for _ in range(N+1)], 'input': [None for _ in range(N+1)], 'rate': [None for _ in range(N)]}
    for k in range(N):
        tar_sym_costs['state'][k] = ca.Function(f'state_stage_{k}', [sym_q], [tar_sym_state_stage])
        tar_sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [tar_sym_input_stage])
        tar_sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [tar_sym_rate_stage])
        tar_sym_costs['state'][N] = ca.Function('state_term', [sym_q], [tar_sym_state_term])
    tar_sym_costs['input'][N] = ca.Function('input_term', [sym_u], [tar_sym_input_term])

    tar_sym_constrs = {'state_input': [None for _ in range(N+1)], 
                        'rate': [None for _ in range(N)]}

    ego_mpcc_controller = CA_MPCC_conv(ego_dynamics_model, 
                                        ego_sym_costs, 
                                        ego_sym_constrs, 
                                        {'qu_ub': ego_state_input_ub, 'qu_lb': ego_state_input_lb, 'du_ub': ego_input_rate_ub, 'du_lb': ego_input_rate_lb},
                                        ego_mpc_params)

    tar_mpcc_controller = CA_MPCC_conv(tar_dynamics_model, 
                                    tar_sym_costs, 
                                    tar_sym_constrs, 
                                    {'qu_ub': tar_state_input_ub, 'qu_lb': tar_state_input_lb, 'du_ub': tar_input_rate_ub, 'du_lb': tar_input_rate_lb},
                                    tar_mpc_params)

    ego_sim_state = scenario.ego_init_state.copy()
    tar_sim_state = scenario.tar_init_state.copy()
    ego_sim_state.t = t
    tar_sim_state.t = t
    track_obj.local_to_global_typed(ego_sim_state)
    track_obj.local_to_global_typed(tar_sim_state)

    ego_u_ws = 0.01*np.ones((N+1, ego_dynamics_model.n_u))
    ego_vs_ws = ego_sim_state.v.v_long*np.ones(N+1)
    ego_du_ws = np.zeros((N, ego_dynamics_model.n_u))
    ego_dvs_ws = np.zeros(N)
    ego_P = np.repeat(np.array([tar_sim_state.x.x, tar_sim_state.x.y]), N)
    ego_R = np.array([])
    ego_mpcc_controller.set_warm_start(ego_u_ws, ego_vs_ws, ego_du_ws, ego_dvs_ws, 
                                       state=ego_sim_state,
                                       reference=ego_R,
                                       parameters=ego_P)

    tar_u_ws = 0.01*np.ones((N+1, tar_dynamics_model.n_u))
    tar_vs_ws = tar_sim_state.v.v_long*np.ones(N+1)
    tar_du_ws = np.zeros((N, tar_dynamics_model.n_u))
    tar_dvs_ws = np.zeros(N)
    tar_P = np.array([0.1, 0.1])
    tar_R = np.zeros(N+1)
    tar_mpcc_controller.set_warm_start(tar_u_ws, tar_vs_ws, tar_du_ws, tar_dvs_ws, 
                                       state=tar_sim_state,
                                       reference=tar_R,
                                       parameters=tar_P)

    egost_list = [ego_sim_state.copy()]
    tarst_list = [tar_sim_state.copy()]

    egopred_list = [None]
    tarpred_list = [None]

    ego_prediction, tar_prediction, tv_pred = None, None, None
    while ego_sim_state.t < T:
        if track_obj.global_to_local_typed(ego_sim_state):
            print('Ego outside of track, stopping...')
            break
        if track_obj.global_to_local_typed(tar_sim_state):
            print('Target outside of track, stopping...')
            break

        if tar_sim_state.p.s >= 1.9 * scenario.length or ego_sim_state.p.s >= 1.9 * scenario.length:
            break
        else:
            # Target agent
            if tar_sim_state.p.s > ego_sim_state.p.s:
                contouring_cost = 20 / (1 + (tar_sim_state.p.s - ego_sim_state.p.s))**2
                tar_P = np.array([contouring_cost, contouring_cost])
                tar_R = ego_sim_state.p.x_tran*np.ones(N+1)
            else:
                tar_P = np.array([0.1, 0.1])
                tar_R = np.array([])
            tar_mpcc_controller.step(tar_sim_state,
                                    reference=tar_R,
                                    parameters=tar_P)
            tar_prediction = tar_mpcc_controller.get_prediction().copy()
            tar_prediction.t = tar_sim_state.t

            # Ego agent
            ego_P = np.array([tar_prediction.x[1:], tar_prediction.y[1:]]).T.ravel()
            ego_R = np.array([])
            ego_mpcc_controller.step(ego_sim_state,
                                     reference=ego_R,
                                     parameters=ego_P)
            ego_prediction = ego_mpcc_controller.get_prediction().copy()
            ego_prediction.t = ego_sim_state.t

            # step forward
            tar_dynamics_simulator.step(tar_sim_state, T=dt)
            ego_dynamics_simulator.step(ego_sim_state, T=dt)

            # log states
            egost_list.append(ego_sim_state.copy())
            tarst_list.append(tar_sim_state.copy())
            egopred_list.append(ego_prediction)
            tarpred_list.append(tar_prediction)
            print(f"Current time: {round(ego_sim_state.t, 2)}")

    scenario_sim_data = SimData(scenario, len(egost_list), egost_list, tarst_list, egopred_list, tarpred_list)

    # pickle_write(scenario_sim_data, os.path.join(gp_dir, 'testcurve.pkl'))
    smoothPlotResults(scenario_sim_data, speedup=0.5, close_loop=False)


if __name__ == '__main__':
    main()