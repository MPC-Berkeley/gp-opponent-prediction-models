#!/usr/bin/env python3
import os
import pickle
import pathlib
import copy
import array
import pdb

import numpy as np
import casadi as ca

import multiprocessing as mp
from collections import deque

from barcgp.common.utils.file_utils import *
from barcgp.common.pytypes import VehicleActuation, VehicleState, BodyLinearVelocity, ParametricPose, VehiclePrediction
from barcgp.common.utils.scenario_utils import SimData, EvalData, smoothPlotResults, ScenarioGenParams, ScenarioGenerator
from barcgp.common.tracks.track import get_track

from barcgp.h2h_configs import dt, N, width
from barcgp.h2h_configs import ego_dynamics_config, ego_sim_dynamics_config, tar_dynamics_config, tar_sim_dynamics_config
from barcgp.h2h_configs import ego_state_input_ub, ego_state_input_lb, ego_input_rate_ub, ego_input_rate_lb
from barcgp.h2h_configs import tar_state_input_ub, tar_state_input_lb, tar_input_rate_ub, tar_input_rate_lb
from barcgp.h2h_configs import ego_mpc_params, tar_mpc_params, pred_mpc_params
from barcgp.h2h_configs import tar_contouring_nominal, ego_inner_collision_quad, ego_outer_collision_quad
from barcgp.h2h_configs import ego_inner_radius, ego_outer_radius, tar_radius
from barcgp.h2h_configs import ego_L, ego_W, tar_L, tar_W

from barcgp.controllers.CA_MPCC_conv import CA_MPCC_conv

from barcgp.dynamics.models.dynamics_models import CasadiDynamicBicycle
from barcgp.simulation.dynamics_simulator import DynamicsSimulator

from barcgp.prediction.trajectory_predictor import ConstantVelocityPredictor, ConstantAngularVelocityPredictor, GPPredictor, NoPredictor, MPCPredictor, CAMPCCPredictor

# Number of evaluation runs for each predictor
total_runs = 100
# total_runs = 1

T = 40

M = 10
policy_name = 'aggressive_blocking_ca'
policy_dir = os.path.join(eval_dir, policy_name)

track_obj = get_track('Monza_Track')

tar_blocking_weight = 20

ego_state_input_ub.x.x = track_obj.track_extents['x_max']+1
ego_state_input_ub.x.y = track_obj.track_extents['y_max']+1
ego_state_input_lb.x.x = track_obj.track_extents['x_min']-1
ego_state_input_lb.x.y = track_obj.track_extents['y_min']-1

tar_state_input_ub.x.x = track_obj.track_extents['x_max']+1
tar_state_input_ub.x.y = track_obj.track_extents['y_max']+1
tar_state_input_lb.x.x = track_obj.track_extents['x_min']-1
tar_state_input_lb.x.y = track_obj.track_extents['y_min']-1

ego_dynamics_model = CasadiDynamicBicycle(0, ego_dynamics_config, track=track_obj)
tar_dynamics_model = CasadiDynamicBicycle(0, tar_dynamics_config, track=track_obj)

pred_bounds = {'qu_ub': tar_state_input_ub, 'qu_lb': tar_state_input_lb, 'du_ub': tar_input_rate_ub, 'du_lb': tar_input_rate_lb}

predictors = [
                GPPredictor(N, None, policy_name, True, M, cov_factor=2),
                GPPredictor(N, None, policy_name, True, M, cov_factor=1),
                GPPredictor(N, None, policy_name, True, M, cov_factor=0.5),
                ConstantAngularVelocityPredictor(N, cov=.01),
                ConstantAngularVelocityPredictor(N, cov=.005),
                ConstantAngularVelocityPredictor(N, cov=.0),
                CAMPCCPredictor(N, None, cov=.01, dynamics_model=tar_dynamics_model, bounds=pred_bounds, mpc_params=pred_mpc_params),
                CAMPCCPredictor(N, None, cov=.005, dynamics_model=tar_dynamics_model, bounds=pred_bounds, mpc_params=pred_mpc_params),
                CAMPCCPredictor(N, None, cov=.0, dynamics_model=tar_dynamics_model, bounds=pred_bounds, mpc_params=pred_mpc_params)
            ]
names = ["GP_2", "GP_1", "GP_05 ", "CAV_01", "CAV_005", "CAV_0", "NLMPC_01", "NLMPC_005", "NLMPC_0"]

# predictors = [
#                 GPPredictor(N, None, policy_name, True, M, cov_factor=2),
#             ]
# names = ["test"]


def main(args=None):

    t = 0  # Initial time increment
    '''
    Collect the actual data
    '''
    ep = 5*np.pi/180
    tarMin = VehicleState(t=0.0,
                        p=ParametricPose(s=0.9, x_tran=-.9 * width/2, e_psi=-ep),
                        v=BodyLinearVelocity(v_long=1.0))
    tarMax = VehicleState(t=0.0,
                        p=ParametricPose(s=1.2, x_tran=.9 * width/2, e_psi=ep),
                        v=BodyLinearVelocity(v_long=1.01))
    egoMin = VehicleState(t=0.0,
                        p=ParametricPose(s=0.2, x_tran=-.9 * width/2, e_psi=-ep),
                        v=BodyLinearVelocity(v_long=1.0))
    egoMax = VehicleState(t=0.0,
                        p=ParametricPose(s=0.4, x_tran=.9 * width/2, e_psi=ep),
                        v=BodyLinearVelocity(v_long=1.01))
    scen_params = ScenarioGenParams(types=['track'], egoMin=egoMin, egoMax=egoMax, tarMin=tarMin,
                                    tarMax=tarMax,
                                    width=width)
    scen_gen = ScenarioGenerator(scen_params)

    gp_params = []
    params = []
    for i in range(total_runs):
        # Set bounds for the sampling of the initial condition
        ego_offset = np.random.uniform(0, 30-5)
        tar_offset = ego_offset + np.random.uniform(5, 10)
        scen_gen.egoMax.p.s += ego_offset
        scen_gen.egoMin.p.s += ego_offset
        scen_gen.tarMax.p.s += tar_offset
        scen_gen.tarMin.p.s += tar_offset
        scen = scen_gen.genScenario()
        for k in range(len(names)):
            predictors[k].track = scen.track
            if isinstance(predictors[k], GPPredictor):
                gp_params.append((dt, t, N, names[k], predictors[k], scen, i, ego_offset))
            else:
                params.append((dt, t, N, names[k], predictors[k], scen, i, ego_offset))

    print("Starting non-GP Evaluation!")
    for p in params:
        runSimulation(*p)

    # process_pool = mp.Pool(processes=10)
    # process_pool.starmap(runSimulation, params)
    # print("non-GP Evaluation Done!")
    # process_pool.close()
    # process_pool.join()

    print("Starting GP Evaluation!")
    for p in gp_params:
        runSimulation(*p)
    print("GP Evaluation Done!")

def runSimulation(dt, t, N, name, predictor, scenario, id, offset=0):
    print(f'{name}, sim {id}')
    tv_inf, ego_inf = False, False
    track_obj = scenario.track

    ego_dynamics_simulator = DynamicsSimulator(t, ego_dynamics_config, track=track_obj)
    tar_dynamics_simulator = DynamicsSimulator(t, tar_dynamics_config, track=track_obj)

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
    sym_p_obs = ca.MX.sym('p_obs', 5)
    p_obs_dict = ['x_ob', 'y_ob', 'psi_ob', 'std_x', 'std_y']

    r_d = ego_W
    dl = ego_L * 0.9
    x_ob = sym_p_obs[p_obs_dict.index('x_ob')]
    y_ob = sym_p_obs[p_obs_dict.index('y_ob')]
    psi_ob = sym_p_obs[p_obs_dict.index('psi_ob')]
    s_e = ca.sin(sym_q[psi_idx])
    c_e = ca.cos(sym_q[psi_idx])
    s = ca.sin(psi_ob)
    c = ca.cos(psi_ob)
    dx1 = x_ob - (sym_q[x_idx] - 3 * dl * c_e / 2)
    dx2 = x_ob - (sym_q[x_idx] - dl * c_e / 2)
    dx3 = x_ob - (sym_q[x_idx] + dl * c_e / 2)
    dx4 = x_ob - (sym_q[x_idx] + 3 * dl * c_e / 2)
    dy1 = y_ob - (sym_q[y_idx] - 3 * dl * s_e / 2)
    dy2 = y_ob - (sym_q[y_idx] - dl * s_e / 2)
    dy3 = y_ob - (sym_q[y_idx] + dl * s_e / 2)
    dy4 = y_ob - (sym_q[y_idx] + 3 * dl * s_e / 2)

    std_x = p_obs_dict.index('std_x')
    std_y = p_obs_dict.index('std_y')
    slack_l_hard = 0.4 * ego_L
    slack_w_hard = 0.4 * ego_W
    slack_l = ca.if_else(std_x == 0, slack_l_hard, sym_p_obs[std_x] + slack_l_hard)
    slack_w = ca.if_else(std_y == 0, slack_w_hard, sym_p_obs[std_y] + slack_w_hard)
    a = (ego_L / np.sqrt(2) + r_d + slack_l)
    b = (ego_W / np.sqrt(2) + r_d + slack_w)

    i1 = (c * dx1 - s * dy1) ** 2 * 1 / (a ** 2 + 0.001) + (s * dx1 + c * dy1) ** 2 * 1 / (b ** 2 + 0.001)
    i2 = (c * dx2 - s * dy2) ** 2 * 1 / (a ** 2 + 0.001) + (s * dx2 + c * dy2) ** 2 * 1 / (b ** 2 + 0.001)
    i3 = (c * dx3 - s * dy3) ** 2 * 1 / (a ** 2 + 0.001) + (s * dx3 + c * dy3) ** 2 * 1 / (b ** 2 + 0.001)
    i4 = (c * dx4 - s * dy4) ** 2 * 1 / (a ** 2 + 0.001) + (s * dx4 + c * dy4) ** 2 * 1 / (b ** 2 + 0.001)

    obs_avoid = (ego_inner_radius+tar_radius)**2 - ca.bilin(ca.DM.eye(2), sym_p-sym_p_obs[:2], sym_p-sym_p_obs[:2])
    
    # f_obs_avoid = ca.Function('obs_avoid', [sym_q, sym_u, sym_p_obs], [ca.vertcat(obs_avoid, 1-i1, 1-i2, 1-i3, 1-i4)])
    f_obs_avoid = ca.Function('obs_avoid', [sym_q, sym_u, sym_p_obs], [ca.vertcat(obs_avoid, 1-i3)])

    ego_sym_constrs = {'state_input': [None] + [f_obs_avoid for _ in range(1, N+1)], 
                'rate': [None for _ in range(N)]}
    
    # ego_mpc_params.soft_constraint_idxs = [[]] + [[0, 1, 2, 3, 4] for _ in range(N)]
    # ego_mpc_params.soft_constraint_quad = [[]] + [[ego_inner_collision_quad, ego_outer_collision_quad, ego_outer_collision_quad, ego_outer_collision_quad, ego_outer_collision_quad] for _ in range(N)]
    # ego_mpc_params.soft_constraint_lin = [[]] + [[0, 0, 0, 0, 0] for _ in range(N)]
    ego_mpc_params.soft_constraint_idxs = [[]] + [[0, 1] for _ in range(N)]
    ego_mpc_params.soft_constraint_quad = [[]] + [[ego_inner_collision_quad, ego_outer_collision_quad] for _ in range(N)]
    ego_mpc_params.soft_constraint_lin = [[]] + [[0, 0] for _ in range(N)]

    ego_mpc_params.performance_cost = 0.1
    ego_mpcc_controller = CA_MPCC_conv(ego_dynamics_model, 
                                        ego_sym_costs, 
                                        ego_sym_constrs, 
                                        {'qu_ub': ego_state_input_ub, 'qu_lb': ego_state_input_lb, 'du_ub': ego_input_rate_ub, 'du_lb': ego_input_rate_lb},
                                        ego_mpc_params)
    
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
    ego_P = np.repeat(np.array([tar_sim_state.x.x, tar_sim_state.x.y, tar_sim_state.e.psi, 0.01, 0.01]), N)
    ego_R = np.array([])
    ego_mpcc_controller.set_warm_start(ego_u_ws, ego_vs_ws, ego_du_ws, ego_dvs_ws, 
                                       state=ego_sim_state,
                                       reference=ego_R,
                                       parameters=ego_P)
    
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
    
    if isinstance(predictor, CAMPCCPredictor):
        predictor.set_warm_start(tar_sim_state)

    egost_list = [ego_sim_state.copy()]
    tarst_list = [tar_sim_state.copy()]

    gp_tarpred_list = [None]
    egopred_list = [None]
    tarpred_list = [None]

    ego_pred_prev = None

    ego_prediction, tar_prediction, tv_pred = None, None, None

    ego_lap = 0
    ego_sim_state_prev = copy.deepcopy(ego_sim_state)
    
    # while True:
    while ego_sim_state.t < T :
        if track_obj.global_to_local_typed(ego_sim_state):
            print(f'Sim {id}: Ego outside of track, stopping...')
            break
        if track_obj.global_to_local_typed(tar_sim_state):
            print(f'Sim {id}: Target outside of track, stopping...')
            break
        
        # ego_speed = ego_sim_state.v.v_long
        # tar_speed = tar_sim_state.v.v_long
        # print(f'Ego speed: {ego_speed} | Tar speed: {tar_speed}')

        if ego_sim_state_prev.p.s - ego_sim_state.p.s > track_obj.track_length/2:
            print(f'Lap {ego_lap} finished')
            ego_lap += 1
            if ego_lap > 1:
                print('Laps completed')
                break
            
        if ego_sim_state.t > T:
            print('Max simulation time reached')
            break

        ego_sim_state_prev = copy.deepcopy(ego_sim_state)

        ego_pred = ego_mpcc_controller.get_prediction()

        if predictor and ego_pred.x and ego_pred.y and ego_pred.psi:
            ego_state = copy.deepcopy(ego_sim_state)
            tar_state = copy.deepcopy(tar_sim_state)

            if ego_state.p.s - tar_state.p.s > track_obj.track_length/2:
                tar_state.p.s += track_obj.track_length

            s_prev = None
            s_pred = []
            ey_pred = []
            ep_pred = []

            for idx in range(len(ego_pred.x)):
                _x, _y, _psi = ego_pred.x[idx], ego_pred.y[idx], ego_pred.psi[idx]
                
                ql = track_obj.global_to_local((_x, _y, _psi))
                if ql is None:
                    s_pred = np.append(ego_pred_prev.s[1:], ego_pred_prev.s[-1])
                    if s_pred[0] - ego_state.p.s > track_obj.track_length/2:
                        s_pred -= track_obj.track_length
                    ey_pred = np.append(ego_pred_prev.x_tran[1:], ego_pred_prev.x_tran[-1])
                    ep_pred = np.append(ego_pred_prev.e_psi[1:], ego_pred_prev.e_psi[-1])
                    break
                else:
                    _s, _ey, _ep = ql

                if idx == 0:
                    if _s - ego_state.p.s > track_obj.track_length/2:
                        _s -= track_obj.track_length
                    if ego_state.p.s - _s > track_obj.track_length/2:
                        _s += track_obj.track_length
                    s_prev = copy.copy(_s)
                    
                if s_prev - _s > track_obj.track_length/2:
                    _s += track_obj.track_length
                s_prev = copy.copy(_s)

                s_pred.append(_s)
                ey_pred.append(_ey)
                ep_pred.append(_ep)
            
            ego_pred.s = array.array('d', s_pred)
            ego_pred.x_tran = array.array('d', ey_pred)
            ego_pred.e_psi = array.array('d', ep_pred)
            
            ego_pred_prev = copy.deepcopy(ego_pred)

            opp_pred = predictor.get_prediction(ego_state, tar_state, ego_pred, tar_prediction)
            if opp_pred.x is None:
                x_pred = []
                y_pred = []
                p_pred = []
                for _s, _ey, _ep in zip(opp_pred.s, opp_pred.x_tran, opp_pred.e_psi):
                    _x, _y, _p = track_obj.local_to_global((_s, _ey, _ep))
                    x_pred.append(_x)
                    y_pred.append(_y)
                    p_pred.append(_p)
                opp_pred.x = array.array('d', x_pred)
                opp_pred.y = array.array('d', y_pred)
                opp_pred.psi = array.array('d', p_pred)
        else:
            opp_pred = None

        gp_tarpred_list.append(copy.deepcopy(opp_pred))
        
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
        if opp_pred is None:
            ego_P = np.repeat(np.array([tar_sim_state.x.x, tar_sim_state.x.y, tar_sim_state.e.psi, 0.01, 0.01]), N)
            ego_R = np.array([])
        else:
            xy_cov = np.array(opp_pred.xy_cov).reshape(-1, 4).reshape(-1, 2, 2)
            x_std = np.sqrt(xy_cov[:,0,0])
            y_std = np.sqrt(xy_cov[:,1,1])
            ego_P = np.array([opp_pred.x, opp_pred.y, opp_pred.psi, x_std, y_std]).T[1:].ravel()
            ego_R = np.array([])
        ego_mpcc_controller.step(ego_sim_state,
                                    reference=ego_R,
                                    parameters=ego_P)
        ego_prediction = ego_mpcc_controller.get_prediction().copy()
        ego_prediction.t = ego_sim_state.t

        # step forward
        tar_dynamics_simulator.step(tar_sim_state, T=dt)
        ego_dynamics_simulator.step(ego_sim_state, T=dt)

        track_obj.update_curvature(tar_sim_state)

        # log states
        egost_list.append(ego_sim_state.copy())
        tarst_list.append(tar_sim_state.copy())
        egopred_list.append(ego_prediction)
        tarpred_list.append(tar_prediction)
        # print('Current time', ego_sim_state.t)

    scenario_sim_data = EvalData(scenario, len(egost_list), egost_list, tarst_list, egopred_list, tarpred_list, gp_tarpred_list, tv_infeasible=tv_inf, ego_infeasible=ego_inf)
    
    root_dir = os.path.join(policy_dir, scenario.track_type)
    create_dir(path=root_dir)
    root_dir = os.path.join(root_dir, name)
    create_dir(path=root_dir)
    pickle_write(scenario_sim_data, os.path.join(root_dir, str(id) + '.pkl'))
    
    # smoothPlotResults(scenario_sim_data, speedup=1.0, close_loop=False)
    # pdb.set_trace()


if __name__ == '__main__':
    main()
