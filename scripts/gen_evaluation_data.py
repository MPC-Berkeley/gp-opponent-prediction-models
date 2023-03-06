#!/usr/bin/env python3
import os
import pickle
import pathlib

import numpy as np
from collections import deque

from barcgp.common.utils.file_utils import *

from barcgp.common.pytypes import VehicleActuation, VehicleState, BodyLinearVelocity, ParametricPose, VehiclePrediction
from barcgp.dynamics.models.model_types import DynamicBicycleConfig
from barcgp.common.utils.scenario_utils import SimData, EvalData, smoothPlotResults, ScenarioGenParams, ScenarioGenerator

from barcgp.controllers.MPCC_H2H_approx import MPCC_H2H_approx

from barcgp.simulation.dynamics_simulator import DynamicsSimulator

import multiprocessing as mp
from barcgp.h2h_configs import *
from barcgp.prediction.trajectory_predictor import ConstantVelocityPredictor, ConstantAngularVelocityPredictor, GPPredictor, NoPredictor, NLMPCPredictor, MPCPredictor
from barcgp.common_control import run_pid_warmstart

total_runs = 100
M = 50
policy_name = 'aggressive_blocking'
policy_dir = os.path.join(eval_dir, policy_name)

predictors = [GPPredictor(N, None, policy_name, True, M, cov_factor=np.sqrt(2)),
                GPPredictor(N, None, policy_name, True, M, cov_factor=1),
                    GPPredictor(N, None, policy_name, True, M, cov_factor=np.sqrt(0.5)),
                ConstantAngularVelocityPredictor(N, cov=.01),
                ConstantAngularVelocityPredictor(N, cov=.005),
                ConstantAngularVelocityPredictor(N, cov=.0),
                NLMPCPredictor(N, None, cov=.01, v_ref=mpcc_tv_params.vx_max),
                NLMPCPredictor(N, None, cov=.005, v_ref=mpcc_tv_params.vx_max),
                NLMPCPredictor(N, None, cov=.0, v_ref=mpcc_tv_params.vx_max)
              ]

"""ConstantVelocityPredictor(N), ConstantAngularVelocityPredictor(N),
NoPredictor(N), MPCPredictor(N), NLMPCPredictor(N, None)]"""
names = ["GP2", "GP1", "GP_5 ", "CAV_01", "CAV_005", "CAV0", "NLMPC_01", "NLMPC_005", "NLMPC0"]


def main(args=None):

    t = 0  # Initial time increment
    '''
    Collect the actual data
    '''
    scen_params = ScenarioGenParams(types=['track'], egoMin=egoMin, egoMax=egoMax, tarMin=tarMin,
                                    tarMax=tarMax,
                                    width=width)
    scen_gen = ScenarioGenerator(scen_params)
    gp_params = []
    params = []
    d = 0
    for i in range(total_runs):
        scen = scen_gen.genScenario()
        offset = np.random.uniform(0, 30)
        for k in range(len(names)):
            predictors[k].track = scen.track
            if isinstance(predictors[k], GPPredictor):
                gp_params.append((dt, t, N, names[k], predictors[k], scen, i+d, offset))
            else:
                params.append((dt, t, N, names[k], predictors[k], scen, i+d, offset))

    print("Starting non-GP Evaluation!")
    # print(len(params[0]))
    # process_pool = mp.Pool(processes=10)
    # process_pool.starmap(runSimulation, params)
    # print("Closing!")
    # process_pool.close()
    # process_pool.join()
    for p in params:
        runSimulation(*p)
        
    print("Starting GP Evaluation!")
    for p in gp_params:
        runSimulation(*p)

def runSimulation(dt, t, N, name, predictor, scenario, id, offset=0):
    print(f'{name} sim {id}')
    tv_inf, ego_inf = False, False
    track_obj = scenario.track

    ego_dynamics_simulator = DynamicsSimulator(t, ego_dynamics_config, track=track_obj)
    tar_dynamics_simulator = DynamicsSimulator(t, tar_dynamics_config, track=track_obj)
    tv_history, ego_history, vehiclestate_history, ego_sim_state, tar_sim_state, egost_list, tarst_list = run_pid_warmstart(
        scenario, ego_dynamics_simulator, tar_dynamics_simulator, n_iter=n_iter, t=t, offset=offset)

    if isinstance(predictor, GPPredictor):
        mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj, gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name='track')
    else:
        mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj, mpcc_ego_params, name="mpcc_h2h_ego", track_name='track')
    mpcc_ego_controller.initialize()
    mpcc_ego_controller.set_warm_start(*ego_history)

    mpcc_tv_controller = MPCC_H2H_approx(tar_dynamics_simulator.model, track_obj, mpcc_tv_params, name="mpcc_h2h_tv", track_name='track')
    mpcc_tv_controller.initialize()
    mpcc_tv_controller.set_warm_start(*tv_history)

    if isinstance(predictor, NLMPCPredictor):
        # predictor.set_warm_start(*vehiclestate_history)
        predictor.set_warm_start()

    gp_tarpred_list = [None] * n_iter
    egopred_list = [None] * n_iter
    tarpred_list = [None] * n_iter

    ego_prediction, tar_prediction, tv_pred = None, None, None
    while True:
        if tar_sim_state.p.s >= 1.9 * scenario.length - offset or ego_sim_state.p.s >= 1.9 * scenario.length - offset or ego_sim_state.t > 37:
            break
        else:
            if predictor:
                ego_pred = mpcc_ego_controller.get_prediction()
                if ego_pred.x is not None:
                    tv_pred = predictor.get_prediction(ego_sim_state, tar_sim_state, ego_pred, tar_prediction)
                    gp_tarpred_list.append(tv_pred.copy())
                else:
                    gp_tarpred_list.append(None)
            # update control inputs
            info, b, exitflag = mpcc_tv_controller.step(tar_sim_state, tv_state=ego_sim_state, tv_pred=ego_prediction, policy=policy_name)
            if not info["success"]:
                if not exitflag == 0:
                    tv_inf=True
                    pass
            info, b, exitflag = mpcc_ego_controller.step(ego_sim_state, tv_state=tar_sim_state, tv_pred=tv_pred)
            if not info["success"]:
                if not exitflag == 0:
                    ego_inf=True
                    pass

            # step forward
            tar_prediction = None if not mpcc_tv_controller.get_prediction() else mpcc_tv_controller.get_prediction().copy()
            tar_prediction.t = tar_sim_state.t
            tar_dynamics_simulator.step(tar_sim_state)
            track_obj.update_curvature(tar_sim_state)

            ego_prediction = None if not mpcc_ego_controller.get_prediction() else mpcc_ego_controller.get_prediction().copy()
            ego_prediction.t = ego_sim_state.t
            # ego_prediction.xy_cov = np.repeat(np.diag([0.001, 0.001])[np.newaxis, :, :], 11, axis=0)
            ego_dynamics_simulator.step(ego_sim_state)

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


if __name__ == '__main__':
    main()
