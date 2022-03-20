#!/usr/bin/env python3
import copy
import os
import pickle
import gc
import torch, gpytorch
import numpy as np
from collections import deque
from barcgp.common.utils.file_utils import *
from barcgp.common.pytypes import VehicleActuation, VehicleState, BodyLinearVelocity, ParametricPose, VehiclePrediction
from barcgp.dynamics.models.model_types import DynamicBicycleConfig
from barcgp.common.utils.scenario_utils import SimData, smoothPlotResults, ScenarioGenParams, ScenarioGenerator, EvalData, post_gp
from barcgp.controllers.MPCC_H2H_approx import MPCC_H2H_approx
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

    ego_dynamics_simulator = DynamicsSimulator(t, ego_dynamics_config, track=track_obj)
    tar_dynamics_simulator = DynamicsSimulator(t, tar_dynamics_config, track=track_obj)

    tv_history, ego_history, vehiclestate_history, ego_sim_state, tar_sim_state, egost_list, tarst_list = \
        run_pid_warmstart(scenario, ego_dynamics_simulator, tar_dynamics_simulator, n_iter=n_iter, t=t)

    gp_mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj, gp_mpcc_ego_params, name="gp_mpcc_h2h_ego", track_name=track_name)
    gp_mpcc_ego_controller.initialize()

    mpcc_ego_controller = MPCC_H2H_approx(ego_dynamics_simulator.model, track_obj, mpcc_ego_params, name="mpcc_h2h_ego", track_name=track_name)
    mpcc_ego_controller.initialize()
    if predictor_class.__name__ == "GPPredictor":
        mpcc_ego_controller = gp_mpcc_ego_controller

    mpcc_ego_controller.set_warm_start(*ego_history)

    mpcc_tv_params.vectorize_constraints()
    mpcc_tv_controller = MPCC_H2H_approx(tar_dynamics_simulator.model, track_obj, mpcc_tv_params, name="mpcc_h2h_tv", track_name=track_name)
    mpcc_tv_controller.initialize()
    mpcc_tv_controller.set_warm_start(*tv_history)

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
            info, b, exitflag = mpcc_tv_controller.step(tar_sim_state, tv_state=ego_sim_state, tv_pred=ego_prediction, policy=policy_name)
            if not info["success"]:
                print(f"TV infeasible - Exitflag: {exitflag}")
                pass

            # Ego agent
            info, b, exitflag = mpcc_ego_controller.step(ego_sim_state, tv_state=tar_sim_state, tv_pred=tv_pred if use_predictions_from_module else tar_prediction)
            if not info["success"]:
                print(f"EGO infeasible - Exitflag: {exitflag}")
                pass
                # return

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