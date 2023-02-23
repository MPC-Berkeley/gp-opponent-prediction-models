#!/usr/bin/env python3
import copy
import os
import pickle
import gc
import torch, gpytorch
import numpy as np
import casadi as ca
from collections import deque
from barcgp.common.utils.file_utils import *
from barcgp.common.pytypes import VehicleActuation, VehicleState, BodyLinearVelocity, ParametricPose, VehiclePrediction
from barcgp.dynamics.models.model_types import DynamicBicycleConfig
from barcgp.common.utils.scenario_utils import SimData, smoothPlotResults, ScenarioGenParams, ScenarioGenerator, EvalData, post_gp
# from barcgp.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from barcgp.controllers.CA_MPCC_conv import CA_MPCC_conv
from barcgp.simulation.dynamics_simulator import DynamicsSimulator
from barcgp.h2h_configs import *

# All models are needed for pickle loading
from barcgp.prediction.gpytorch_models import MultitaskGPModelApproximate, MultitaskGPModel, \
    IndependentMultitaskGPModelApproximate, ExactGPModel
from barcgp.prediction.gp_controllers import GPControllerTrained
from barcgp.prediction.trajectory_predictor import ConstantVelocityPredictor, ConstantAngularVelocityPredictor, \
    GPPredictor, NLMPCPredictor

from barcgp.common_control import run_pid_warmstart

def main(args=None):
    ##############################################################################################################################################
    use_GPU = True   
    gen_scenario = True  # Controls whether to generate new scenario or use saved pkl
    predictor_class = GPPredictor  # Either None or one of trajectory_predictor classes
    use_predictions_from_module = True  # Set to true to use predictions generated from `predictor_class`, otherwise use true predictions from MPCC
    policy_name = "aggressive_blocking"
    M = 50  # Number of samples for GP
    T = 20  # Max number of seconds to run experiment
    t = 0  # Initial time increment
    ##############################################################################################################################################
    if gen_scenario:
        scen_params = ScenarioGenParams(types=['track'], egoMin=egoMin, egoMax=egoMax, tarMin=tarMin, tarMax=tarMax, width=width)
        scen_gen = ScenarioGenerator(scen_params)
        scenario = scen_gen.genScenario()
    else:
        scenario_sim_data = pickle_read(os.path.join(gp_dir, 'testcurve.pkl'))
        scenario = scenario_sim_data.scenario_def

    track_name = scenario.track_type
    track_obj = scenario.track

    state_input_ub_ego=VehicleState(x=Position(x=track_obj.track_extents['x_max']+1, y=track_obj.track_extents['y_max']+1),
                    e=OrientationEuler(psi=100),
                    v=BodyLinearVelocity(v_long=0.5*factor))
    state_input_lb_ego=VehicleState(x=Position(x=track_obj.track_extents['x_min']-1, y=track_obj.track_extents['y_min']-1),
                                e=OrientationEuler(psi=-100),
                                v=BodyLinearVelocity(v_long=0.5*factor))
    input_rate_ub_ego = VehicleState(u=VehicleActuation(u_a=1, u_steer=0.5))
    input_rate_lb_ego = VehicleState(u=VehicleActuation(u_a=-1, u_steer=-0.5))

    state_input_ub_tv=VehicleState(x=Position(x=track_obj.track_extents['x_max']+1, y=track_obj.track_extents['y_max']+1),
                            e=OrientationEuler(psi=100),
                            v=BodyLinearVelocity(v_long=0.5*factor))
    state_input_lb_tv=VehicleState(x=Position(x=track_obj.track_extents['x_min']-1, y=track_obj.track_extents['y_min']-1),
                                e=OrientationEuler(psi=-100),
                                v=BodyLinearVelocity(v_long=0.5*factor))
    input_rate_ub_tv = VehicleState(u=VehicleActuation(u_a=1, u_steer=0.5))
    input_rate_lb_tv = VehicleState(u=VehicleActuation(u_a=-1, u_steer=-0.5))

    ego_dynamics_simulator = DynamicsSimulator(t, ego_dynamics_config, track=track_obj)
    tar_dynamics_simulator = DynamicsSimulator(t, tar_dynamics_config, track=track_obj)

    tv_history, ego_history, vehiclestate_history, ego_sim_state, tar_sim_state, egost_list, tarst_list = \
        run_pid_warmstart(scenario, ego_dynamics_simulator, tar_dynamics_simulator, n_iter=n_iter, t=t)

    sym_q = ca.MX.sym('q', ego_dynamics_simulator.model.n_q)
    sym_u = ca.MX.sym('u', ego_dynamics_simulator.model.n_u)
    sym_du = ca.MX.sym('du', ego_dynamics_simulator.model.n_u)

    vy_idx = 1
    wz_idx = 2
    x_idx = 3
    y_idx = 4
    psi_idx = 5

    ua_idx = 0
    us_idx = 1

    sym_state_stage = sym_state_term = 0
    sym_input_stage = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
    sym_input_term = 0.5*(1e-4*(sym_u[ua_idx])**2 + 1e-4*(sym_u[us_idx])**2)
    sym_rate_stage = 0.5*(0.01*(sym_du[ua_idx])**2 + 0.01*(sym_du[us_idx])**2)

    sym_costs = {'state': [None for _ in range(N+1)], 'input': [None for _ in range(N+1)], 'rate': [None for _ in range(N)]}
    for k in range(N):
        sym_costs['state'][k] = ca.Function(f'state_stage_{k}', [sym_q], [sym_state_stage])
        sym_costs['input'][k] = ca.Function(f'input_stage_{k}', [sym_u], [sym_input_stage])
        sym_costs['rate'][k] = ca.Function(f'rate_stage_{k}', [sym_du], [sym_rate_stage])
        sym_costs['state'][N] = ca.Function('state_term', [sym_q], [sym_state_term])
    sym_costs['input'][N] = ca.Function('input_term', [sym_u], [sym_input_term])

    a_max = 0.9*ego_dynamics_config.gravity*ego_dynamics_config.wheel_friction
    sym_ax, sym_ay, _ = ego_dynamics_simulator.model.f_a(sym_q, sym_u)
    friction_circle_constraint = ca.Function('friction_circle', [sym_q, sym_u], [sym_ax**2 + sym_ay**2 - a_max**2])

    sym_p = sym_q[[x_idx, y_idx]]
    sym_p_obs = ca.MX.sym('p_obs', 2)
    soft_vehicle_radius = 0.12
    opponent_radius = 0.12
    soft_obs_avoid = (soft_vehicle_radius+opponent_radius)**2 - ca.bilin(ca.DM.eye(2), sym_p-sym_p_obs, sym_p-sym_p_obs)
    sym_constrs = {'state_input': [None for _ in range(N+1)], 
                    'rate': [None for _ in range(N)]}
    # sym_constrs = {'state_input': [None] + [f_obs_avoid for _ in range(1, N+1)], 
                        # 'rate': [None for _ in range(N)]}
            
    # mpcc_ego_params.soft_constraint_idxs = [[]] + [[0, 1] for _ in range(N)]
    # mpcc_ego_params.soft_constraint_quad = [[]] + [[100, 40] for _ in range(N)]
    # mpcc_ego_params.soft_constraint_lin = [[]] + [[0, 0] for _ in range(N)]

    # mpcc_tv_params.soft_constraint_idxs = [[]] + [[0, 1] for _ in range(N)]
    # mpcc_tv_params.soft_constraint_quad = [[]] + [[100, 40] for _ in range(N)]
    # mpcc_tv_params.soft_constraint_lin = [[]] + [[0, 0] for _ in range(N)]
    
    mpcc_ego_controller = CA_MPCC_conv(ego_dynamics_simulator.model, 
                                        sym_costs, 
                                        sym_constrs, 
                                        {'qu_ub': state_input_ub_ego, 'qu_lb': state_input_lb_ego, 'du_ub': input_rate_ub_ego, 'du_lb': input_rate_lb_ego},
                                        mpcc_ego_params)

    mpcc_tv_controller = CA_MPCC_conv(ego_dynamics_simulator.model, 
                                    sym_costs, 
                                    sym_constrs, 
                                    {'qu_ub': state_input_ub_tv, 'qu_lb': state_input_lb_tv, 'du_ub': input_rate_ub_tv, 'du_lb': input_rate_lb_tv},
                                    mpcc_tv_params)


    # if predictor_class.__name__ == "GPPredictor":
        # mpcc_ego_controller = gp_mpcc_ego_controller
    gp_mpcc_ego_params = mpcc_ego_params

    # mpcc_ego_controller.set_warm_start(*ego_history)
    # mpcc_tv_params.vectorize_constraints()
    # mpcc_tv_controller = MPCC_H2H_approx(tar_dynamics_simulator.model, track_obj, mpcc_tv_params, name="mpcc_h2h_tv", track_name=track_name)
    # mpcc_tv_controller.initialize()
    # mpcc_tv_controller.set_warm_start(*tv_history)

    ego_history_actuation = VehicleActuation()
    tv_history_actuation = VehicleActuation()
    ego_dynamics_simulator.model.u2input(ego_history_actuation, ego_history[0][-1])
    ego_dynamics_simulator.model.u2input(tv_history_actuation, tv_history[0][-1])

    u_ws = np.tile(ego_dynamics_simulator.model.input2u(ego_history_actuation), (N+1, 1))
    # vs_ws = np.zeros(self.N+1)
    vs_ws = ego_sim_state.v.v_long*np.ones(N+1)
    du_ws = np.zeros((N, ego_dynamics_simulator.model.n_u))
    dvs_ws = np.zeros(N)
    print(u_ws)
    mpcc_ego_controller.set_warm_start(u_ws, vs_ws, du_ws, dvs_ws, ego_sim_state)

    u_ws2 = np.tile(ego_dynamics_simulator.model.input2u(tv_history_actuation), (N+1, 1))
    # vs_ws = np.zeros(self.N+1)
    vs_ws2 = tar_sim_state.v.v_long*np.ones(N+1)
    du_ws2 = np.zeros((N, ego_dynamics_simulator.model.n_u))
    dvs_ws2 = np.zeros(N)
    mpcc_ego_controller.set_warm_start(u_ws2, vs_ws2, du_ws2, dvs_ws2, tar_sim_state)

    predictor = None
    if predictor_class is not None:
        if predictor_class.__name__ == "GPPredictor":
            predictor = GPPredictor(N=N, track=track_obj, policy_name=policy_name, use_GPU=use_GPU, M=M, cov_factor=np.sqrt(2))
        elif predictor_class.__name__ == "NLMPCPredictor":
            predictor = NLMPCPredictor(N=N, track=track_obj, cov=0.01, v_ref=mpcc_tv_params.vx_max)
            predictor.set_warm_start()
        else:
            predictor = predictor_class(N=N, track=track_obj, cov = 0.01)

    gp_tarpred_list = [None] * n_iter
    egopred_list = [None] * n_iter
    tarpred_list = [None] * n_iter

    ego_prediction, tar_prediction, tv_pred = None, None, None
    while ego_sim_state.t < T:
        if tar_sim_state.p.s >= 1.9 * scenario.length or ego_sim_state.p.s >= 1.9 * scenario.length:
            break
        else:
            if predictor:
                ego_pred = mpcc_ego_controller.get_prediction()
                if ego_pred.s is not None:
                    tv_pred = predictor.get_prediction(ego_sim_state, tar_sim_state, ego_pred)
                    gp_tarpred_list.append(tv_pred.copy())
                else:
                    gp_tarpred_list.append(None)

            # Target agent
            mpcc_tv_controller.step(tar_sim_state)

            # Ego agent
            mpcc_ego_controller.step(ego_sim_state)

            # step forward
            tar_prediction = mpcc_tv_controller.get_prediction().copy()
            tar_prediction.t = tar_sim_state.t
            tar_dynamics_simulator.step(tar_sim_state)
            track_obj.update_curvature(tar_sim_state)

            ego_prediction = mpcc_ego_controller.get_prediction().copy()
            ego_prediction.t = ego_sim_state.t
            ego_dynamics_simulator.step(ego_sim_state)

            # log states
            egost_list.append(ego_sim_state.copy())
            tarst_list.append(tar_sim_state.copy())
            egopred_list.append(ego_prediction)
            tarpred_list.append(tar_prediction)
            print(f"Current time: {round(ego_sim_state.t, 2)}")


    if predictor_class:
        scenario_sim_data = EvalData(scenario, len(egost_list), egost_list, tarst_list, egopred_list, tarpred_list, gp_tarpred_list)
    else:
        scenario_sim_data = SimData(scenario, len(egost_list), egost_list, tarst_list, egopred_list, tarpred_list)

    pickle_write(scenario_sim_data, os.path.join(gp_dir, 'testcurve.pkl'))
    smoothPlotResults(scenario_sim_data, speedup=1.6, close_loop=False)


if __name__ == '__main__':
    main()