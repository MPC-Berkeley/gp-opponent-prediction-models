#!/usr/bin/env python3

import multiprocessing as mp
from collections import deque

import numpy as np
from barcgp.dynamics.models.model_types import DynamicBicycleConfig
from barcgp.common.pytypes import VehicleActuation, VehicleState, BodyLinearVelocity, ParametricPose
from barcgp.common.utils.file_utils import *
from barcgp.common.utils.scenario_utils import SimData, ScenarioGenParams, ScenarioGenerator
from barcgp.controllers.MPCC_H2H_approx import MPCC_H2H_approx
from barcgp.simulation.dynamics_simulator import DynamicsSimulator
from barcgp.h2h_configs import *
from barcgp.common_control import run_pid_warmstart

total_runs = 500

policy_name = 'aggressive_blocking'
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
    process_pool = mp.Pool(processes=8)
    params = []
    for i in range(total_runs):
        params.append((dt, t, N, scen_gen.genScenario(), i))

    process_pool.starmap(runSimulation, params)
    process_pool.close()
    process_pool.join()


def runSimulation(dt, t, N, scenario, id):
    print('Sim', id)
    track_obj = scenario.track

    ego_dynamics_simulator = DynamicsSimulator(t, ego_dynamics_config, track=track_obj)
    tar_dynamics_simulator = DynamicsSimulator(t, tar_dynamics_config, track=track_obj)

    tv_history, ego_history, _, ego_sim_state, tar_sim_state, egost_list, tarst_list = run_pid_warmstart(
        scenario, ego_dynamics_simulator, tar_dynamics_simulator, n_iter=n_iter, t=t)

    mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj, mpcc_ego_params, name="mpcc_h2h_ego")
    mpcc_ego_controller.initialize()
    mpcc_ego_controller.set_warm_start(*ego_history)

    mpcc_tv_controller = MPCC_H2H_approx(tar_dynamics_simulator.model, track_obj, mpcc_tv_params, name="mpcc_h2h_tv")
    mpcc_tv_controller.initialize()
    mpcc_tv_controller.set_warm_start(*tv_history)

    egopred_list = [None] * n_iter
    tarpred_list = [None] * n_iter

    ego_prediction, tar_prediction = None, None
    done = False
    while ego_sim_state.t < T and not done:
        if tar_sim_state.p.s >= 1.9 * scenario.length or ego_sim_state.p.s >= 1.9 * scenario.length:
            break
        else:
            # update control inputs
            info, b, exitflag = mpcc_tv_controller.step(tar_sim_state, tv_state=ego_sim_state, tv_pred=ego_prediction, policy=policy_name)
            if not info["success"]:
                if not exitflag == 0:
                    print(f"{id} infeasible", exitflag)
                    done = True
                    break
            info, b, exitflag = mpcc_ego_controller.step(ego_sim_state, tv_state=tar_sim_state, tv_pred=tar_prediction)
            if not info["success"]:
                pass
                # print(f"{id} infeasible")
                # return

            # step forward
            tar_prediction = mpcc_tv_controller.get_prediction().copy()
            tar_prediction.t = tar_sim_state.t
            tar_prediction.xy_cov = np.repeat(np.diag([0.001, 0.001])[np.newaxis, :, :], 11, axis=0)
            tar_dynamics_simulator.step(tar_sim_state)
            track_obj.update_curvature(tar_sim_state)

            ego_prediction = mpcc_ego_controller.get_prediction().copy()
            ego_prediction.t = ego_sim_state.t
            ego_prediction.xy_cov = np.repeat(np.diag([0.001, 0.001])[np.newaxis, :, :], 11, axis=0)
            ego_dynamics_simulator.step(ego_sim_state)

            # log states
            egost_list.append(ego_sim_state.copy())
            tarst_list.append(tar_sim_state.copy())
            egopred_list.append(ego_prediction)
            tarpred_list.append(tar_prediction)
            # print('Current time', ego_sim_state.t)

    scenario_sim_data = SimData(scenario, len(egost_list), egost_list, tarst_list, egopred_list, tarpred_list)
    root_dir = os.path.join(policy_dir, scenario.track_type)
    create_dir(path=root_dir)
    pickle_write(scenario_sim_data, os.path.join(root_dir, str(id) + '.pkl'))


if __name__ == '__main__':
    main()