#!/usr/bin/env python3

import multiprocessing as mp
from collections import deque

import numpy as np
import casadi as ca

from barcgp.common.utils.file_utils import *
from barcgp.common.utils.scenario_utils import SimData, smoothPlotResults, ScenarioGenParams, ScenarioGenerator, EvalData, post_gp
from barcgp.common.pytypes import *

from barcgp.h2h_configs import dt, N, width
from barcgp.h2h_configs import ego_dynamics_config, ego_sim_dynamics_config, tar_dynamics_config, tar_sim_dynamics_config
from barcgp.h2h_configs import ego_state_input_ub, ego_state_input_lb, ego_input_rate_ub, ego_input_rate_lb
from barcgp.h2h_configs import tar_state_input_ub, tar_state_input_lb, tar_input_rate_ub, tar_input_rate_lb
from barcgp.h2h_configs import ego_mpc_params, tar_mpc_params
from barcgp.h2h_configs import egoMax, egoMin, tarMax, tarMin
from barcgp.h2h_configs import tar_contouring_nominal, tar_blocking_weight, ego_inner_collision_quad, ego_outer_collision_quad
from barcgp.h2h_configs import ego_inner_radius, ego_outer_radius, tar_radius

from barcgp.controllers.CA_MPCC_conv import CA_MPCC_conv

from barcgp.dynamics.models.dynamics_models import CasadiDynamicBicycle
from barcgp.simulation.dynamics_simulator import DynamicsSimulator

total_runs = 500

policy_name = 'aggressive_blocking_ca'
policy_dir = os.path.join(train_dir, policy_name)
track_types = ['curve', 'chicane']
T = 20

def main(args=None):
    t = 0  # Initial time increment
    '''
    Collect the actual data
    '''
    scen_params = ScenarioGenParams(types=track_types, egoMin=egoMin, egoMax=egoMax, tarMin=tarMin,
                                    tarMax=tarMax,
                                    width=width)
    scen_gen = ScenarioGenerator(scen_params)
        
    params = []
    for i in range(total_runs):
        params.append((dt, t, N, scen_gen.genScenario(), i))

    process_pool = mp.Pool(processes=15)
    process_pool.starmap(runSimulation, params)
    process_pool.close()
    process_pool.join()

    # for p in params:
    #     runSimulation(*p)


def runSimulation(dt, t, N, scenario, id):
    print(f'Starting sim {id}')
    track_obj = scenario.track

    ego_state_input_ub.x.x = track_obj.track_extents['x_max']+1
    ego_state_input_ub.x.y = track_obj.track_extents['y_max']+1
    ego_state_input_lb.x.x = track_obj.track_extents['x_min']-1
    ego_state_input_lb.x.y = track_obj.track_extents['y_min']-1

    tar_state_input_ub.x.x = track_obj.track_extents['x_max']+1
    tar_state_input_ub.x.y = track_obj.track_extents['y_max']+1
    tar_state_input_lb.x.x = track_obj.track_extents['x_min']-1
    tar_state_input_lb.x.y = track_obj.track_extents['y_min']-1

    ego_dynamics_model = CasadiDynamicBicycle(t, ego_dynamics_config, track=track_obj)
    tar_dynamics_model = CasadiDynamicBicycle(t, tar_dynamics_config, track=track_obj)

    ego_dynamics_simulator = DynamicsSimulator(t, ego_sim_dynamics_config, track=track_obj)
    tar_dynamics_simulator = DynamicsSimulator(t, tar_sim_dynamics_config, track=track_obj)

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

    obs_avoid = (ego_inner_radius+tar_radius)**2 - ca.bilin(ca.DM.eye(2), sym_p-sym_p_obs, sym_p-sym_p_obs)
    soft_obs_avoid = (ego_outer_radius+tar_radius)**2 - ca.bilin(ca.DM.eye(2), sym_p-sym_p_obs, sym_p-sym_p_obs)
    
    f_obs_avoid = ca.Function('obs_avoid', [sym_q, sym_u, sym_p_obs], [ca.vertcat(obs_avoid, soft_obs_avoid)])
    ego_sym_constrs = {'state_input': [None] + [f_obs_avoid for _ in range(1, N+1)], 
                'rate': [None for _ in range(N)]}
    
    ego_mpc_params.soft_constraint_idxs = [[]] + [[0, 1] for _ in range(N)]
    ego_mpc_params.soft_constraint_quad = [[]] + [[ego_inner_collision_quad, ego_outer_collision_quad] for _ in range(N)]
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

    # tar_u_ws = np.tile(tar_dynamics_model.input2u(tar_history_actuation), (N+1, 1))
    tar_u_ws = 0.01*np.ones((N+1, tar_dynamics_model.n_u))
    tar_vs_ws = tar_sim_state.v.v_long*np.ones(N+1)
    tar_du_ws = np.zeros((N, tar_dynamics_model.n_u))
    tar_dvs_ws = np.zeros(N)
    tar_P = np.array([tar_contouring_nominal, tar_contouring_nominal])
    tar_R = np.zeros(N+1)
    tar_mpcc_controller.set_warm_start(tar_u_ws, tar_vs_ws, tar_du_ws, tar_dvs_ws, 
                                       state=tar_sim_state,
                                       reference=tar_R,
                                       parameters=tar_P)

    egost_list = [ego_sim_state.copy()]
    tarst_list = [tar_sim_state.copy()]

    egopred_list = [None]
    tarpred_list = [None]

    ego_prediction, tar_prediction = None, None
    done = False
    while ego_sim_state.t < T and not done:
        if track_obj.global_to_local_typed(ego_sim_state):
            print(f'Sim {id}: Ego outside of track, stopping...')
            break
        if track_obj.global_to_local_typed(tar_sim_state):
            print(f'Sim {id}: Target outside of track, stopping...')
            break

        # if tar_sim_state.p.s >= 1.9 * scenario.length or ego_sim_state.p.s >= 1.9 * scenario.length:
        #     break
        if tar_sim_state.p.s >= 1.0 * scenario.length or ego_sim_state.p.s >= 1.0 * scenario.length:
            print(f'Sim {id}: Reached end of track, stopping')
            break
        else:
            # Target agent
            if tar_sim_state.p.s > ego_sim_state.p.s:
                contouring_cost = tar_blocking_weight / (1 + (tar_sim_state.p.s - ego_sim_state.p.s))**2
                tar_P = np.array([contouring_cost, contouring_cost])
                tar_R = ego_sim_state.p.x_tran*np.ones(N+1)
            else:
                tar_P = np.array([tar_contouring_nominal, tar_contouring_nominal])
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
            # print('Current time', ego_sim_state.t)

    scenario_sim_data = SimData(scenario, len(egost_list), egost_list, tarst_list, egopred_list, tarpred_list)
    
    # smoothPlotResults(scenario_sim_data, speedup=0.5, close_loop=False)

    root_dir = os.path.join(policy_dir, scenario.track_type)
    create_dir(path=root_dir)
    pickle_write(scenario_sim_data, os.path.join(root_dir, str(id) + '.pkl'))

if __name__ == '__main__':
    main()