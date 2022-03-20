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
from barcgp.common.utils.scenario_utils import SimData, smoothPlotResults, ScenarioGenParams, ScenarioGenerator, evaluateGP, EvalData, post_gp
from barcgp.controllers.PID import PIDLaneFollower
from barcgp.controllers.utils.controllerTypes import PIDParams
from barcgp.simulation.dynamics_simulator import DynamicsSimulator
from barcgp.h2h_configs import *

# All models are needed for pickle loading
from barcgp.prediction.gpytorch_models import MultitaskGPModelApproximate, MultitaskGPModel, \
    IndependentMultitaskGPModelApproximate, ExactGPModel
from barcgp.prediction.gp_controllers import GPControllerTrained
from barcgp.prediction.trajectory_predictor import ConstantVelocityPredictor, ConstantAngularVelocityPredictor, \
    GPPredictor, NLMPCPredictor
from barcgp.common.utils.scenario_utils import ScenarioDefinition


def run_pid_warmstart(scenario : ScenarioDefinition, ego_dynamics_simulator : DynamicsSimulator, tar_dynamics_simulator : DynamicsSimulator,
                      n_iter, t=0, offset=0, approx=True):
    track_obj = scenario.track

    ego_sim_state = scenario.ego_init_state.copy()
    tar_sim_state = scenario.tar_init_state.copy()
    ego_sim_state.p.s -= offset
    tar_sim_state.p.s -= offset
    track_obj.local_to_global_typed(ego_sim_state)
    track_obj.local_to_global_typed(tar_sim_state)

    track_obj.update_curvature(tar_sim_state)
    state_history_tv = deque([], N); input_history_tv = deque([], N)
    state_history_ego = deque([], N); input_history_ego = deque([], N)
    state_history_vehiclestates = deque([], N+1); input_history_vehiclestates = deque([], N)
    input_ego = VehicleActuation()
    input_tv = VehicleActuation()
    x_ref = ego_sim_state.p.x_tran
    pid_steer_params = PIDParams()
    pid_steer_params.dt = dt
    pid_steer_params.default_steer_params()
    pid_steer_params.Kp = 1
    pid_speed_params = PIDParams()
    pid_speed_params.dt = dt
    pid_speed_params.default_speed_params()
    pid_controller_1 = PIDLaneFollower(ego_sim_state.v.v_long, x_ref, dt, pid_steer_params, pid_speed_params)
    pid_controller_2 = PIDLaneFollower(tar_sim_state.v.v_long, tar_sim_state.p.x_tran, dt, pid_steer_params, pid_speed_params)

    egost_list = [ego_sim_state.copy()]
    tarst_list = [tar_sim_state.copy()]
    while n_iter > 0:
        pid_controller_1.step(ego_sim_state)
        pid_controller_2.step(tar_sim_state)
        ego_dynamics_simulator.step(ego_sim_state)
        tar_dynamics_simulator.step(tar_sim_state)
        input_ego.t = t
        ego_sim_state.copy_control(input_ego)
        q, _ = ego_dynamics_simulator.model.state2qu(ego_sim_state)
        u = ego_dynamics_simulator.model.input2u(input_ego)
        if approx:
            q = np.append(q, ego_sim_state.p.s)
            q = np.append(q, ego_sim_state.p.s)
            u = np.append(u, ego_sim_state.v.v_long)
        state_history_ego.append(q)
        input_history_ego.append(u)

        input_tv.t = t
        tar_sim_state.copy_control(input_tv)
        q, _ = tar_dynamics_simulator.model.state2qu(tar_sim_state)
        u = tar_dynamics_simulator.model.input2u(input_tv)
        track_obj.update_curvature(tar_sim_state)
        if approx:
            q = np.append(q, tar_sim_state.p.s)
            q = np.append(q, tar_sim_state.p.s)
            u = np.append(u, tar_sim_state.v.v_long)
        state_history_tv.append(q)
        input_history_tv.append(u)

        # Save states for nl_mpc TV
        state_history_vehiclestates.append(tar_sim_state.copy())
        input_history_vehiclestates.append(input_tv.copy())

        egost_list.append(ego_sim_state.copy())
        tarst_list.append(tar_sim_state.copy())
        n_iter -= 1
        t += dt

    compose_history = lambda state_history, input_history: (np.array(state_history), np.array(input_history))

    return compose_history(state_history_tv, input_history_tv), compose_history(state_history_ego, input_history_ego), \
           compose_history(state_history_vehiclestates, input_history_vehiclestates), ego_sim_state, tar_sim_state, egost_list, tarst_list