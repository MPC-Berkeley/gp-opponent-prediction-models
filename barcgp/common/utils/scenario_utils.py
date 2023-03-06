#!/usr/bin/env python3
import copy
import os
import pickle
import multiprocessing as mp
import time
import string
from dataclasses import dataclass, field
import random

import pynput
from typing import List, Tuple
import numpy as np
import scipy.interpolate

import matplotlib.pyplot as plt

from barcgp.common.pytypes import VehicleState, VehiclePrediction, ParametricPose, BodyLinearVelocity, VehicleActuation
from barcgp.common.tracks.radius_arclength_track import RadiusArclengthTrack
from barcgp.common.tracks.track_lib import CurveTrack, StraightTrack, ChicaneTrack
from barcgp.common.tracks.track import get_track
from barcgp.visualization.barc_plotter_qt import BarcFigure, GlobalPlotConfigs, VehiclePlotConfigs

@dataclass
class PostprocessData:
    N: int = field(default=None)  # Number of runs evaluated
    setup_id: str = field(default=None)  # name of the setup
    name: str = field(default=None)  # name of the run (GP1, ...)
    # Win metrics
    win_ids: List = field(default_factory=lambda: [])  # names of pickles won
    num_wins: int = field(default=0)
    win_rate: float = field(default=0)  # Win Rate: wins/N
    win_rate_nc: float = field(default=0)  # Win Rate: wins/(N - num_crashes)
    # Crash/Rule violation Metrics
    crash_ids: List = field(default_factory=lambda: [])  # ids of crash pickles
    crash_ids_ego: List = field(default_factory=lambda:[])
    crash_ids_tv: List = field(default_factory=lambda:[])
    num_crashes: int = field(default=0)
    crash_rate: float = field(default=0)
    crash_x: List = field(default_factory=lambda: [])  # Crash positions x
    crash_y: List = field(default_factory=lambda: [])  # Crash positions y
    left_track_ids: List = field(default_factory=lambda: [])
    num_left_track: int = field(default=0)
    left_track_rate: float = field(default=0)
    # Overtake Metrics
    overtake_ids: List = field(default_factory=lambda: [])  # name of overtake pickles
    num_overtakes: int = field(default=0)  # Number of overtakes
    overtake_s: List = field(default_factory=lambda: [])  # Overtake positions s
    overtake_x: List = field(default_factory=lambda: [])  # Overtake positions x
    overtake_y: List = field(default_factory=lambda: [])  # Overtake positions y
    avg_delta_s: float = field(default=0)  # Average terminal delta s
    # Actuation metrics
    avg_a: float = field(default=0)  # Average Acceleration
    avg_min_a: float = field(default=0)  # Average minimum Acceleration per run
    avg_max_a: float = field(default=0)  # Average maximum Acceleration per run
    avg_abs_steer: float = field(default=0)  # Average abs steer value
    # Prediction metrics
    lateral_errors: List = field(default_factory=lambda: [])
    longitudinal_errors: List = field(default_factory=lambda: [])
    # Feasibility Data
    ego_infesible_ids: List = field(default_factory=lambda: [])
    tv_infesible_ids: List = field(default_factory=lambda: [])
    num_ego_inf: int = field(default=0)
    ego_inf_rate: float = field(default=0)
    num_tv_inf: int = field(default=0)
    tv_inf_rate: float = field(default=0)
    # Track
    track: RadiusArclengthTrack = field(default=None)

    def post(self):
        self.win_rate = self.num_wins/self.N
        self.win_rate_nc = self.num_wins/(self.N - self.num_crashes)
        self.crash_rate = self.num_crashes/self.N
        self.left_track_rate = self.num_left_track/self.N
        self.ego_inf_rate = self.num_ego_inf/self.N
        self.tv_inf_rate = self.num_tv_inf / self.N

@dataclass
class ScenarioDefinition:
    track_type: string = field(default=None)
    track: RadiusArclengthTrack = field(default=None)
    ego_init_state: VehicleState = field(default=None)
    tar_init_state: VehicleState = field(default=None)
    ego_obs_avoid_d: float = field(default=None)
    tar_obs_avoid_d: float = field(default=None)
    length: float = field(default=None)

    def __post_init__(self):
        self.length = self.track.track_length
        if self.track.phase_out:
            self.length = self.track.track_length - self.track.cl_segs[-1][0]


@dataclass
class SimData():
    scenario_def: ScenarioDefinition
    N: int
    ego_states: List[VehicleState]
    tar_states: List[VehicleState]
    ego_preds: List[VehiclePrediction] = field(default=List[VehiclePrediction])
    tar_preds: List[VehiclePrediction] = field(default=List[VehiclePrediction])


@dataclass
class EvalData(SimData):
    tar_gp_pred: List[VehiclePrediction] = field(default=List[VehiclePrediction])
    tar_gp_pred_post: List[VehiclePrediction] = field(default=List[VehiclePrediction])
    tv_infeasible: bool = field(default=False)
    ego_infeasible: bool = field(default=False)



@dataclass
class ScenarioGenParams:
    types: list = field(
        default_factory=lambda: ['curve', 'straight', 'chicane'])  # curve, straight, chicane, list of types used
    egoMin: VehicleState = field(default=None)
    egoMax: VehicleState = field(default=None)
    tarMin: VehicleState = field(default=None)
    tarMax: VehicleState = field(default=None)
    width: float = field(default=None)


class ScenarioGenerator:
    def __init__(self, genParams: ScenarioGenParams):
        self.types = genParams.types
        self.egoMin = genParams.egoMin
        self.egoMax = genParams.egoMax
        self.tarMin = genParams.tarMin
        self.tarMax = genParams.tarMax
        self.width = genParams.width

    def genScenarioStatic(self, scenarioType : str, ego_init_state : VehicleState, tar_init_state : VehicleState):
        print('Generating scenario of type:', scenarioType)
        if scenarioType == 'curve':
            return self.genCurve(ego_init_state, tar_init_state)
        elif scenarioType == 'straight':
            return self.genStraight(ego_init_state, tar_init_state)
        elif scenarioType == 'chicane':
            return self.genChicane(ego_init_state, tar_init_state)
        elif scenarioType == 'track':
            return self.getLabTrack(ego_init_state, tar_init_state)

    def genScenario(self):
        """
        Generate random scenario from ego_min, ego_max, tar_min, tar_max
        """
        scenarioType = np.random.choice(self.types)
        s, x_tran, e_psi, v_long = self.randomInit(self.egoMin, self.egoMax)
        ego_init_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))
        s, x_tran, e_psi, v_long = self.randomInit(self.tarMin, self.tarMax)
        while(abs(s-ego_init_state.p.s) < .5 and abs(x_tran - ego_init_state.p.x_tran) < 0.4):
            s, x_tran, e_psi, v_long = self.randomInit(self.egoMin, self.egoMax)
            ego_init_state = VehicleState(t=0.0,
                                          p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                          v=BodyLinearVelocity(v_long=v_long))
            s, x_tran, e_psi, v_long = self.randomInit(self.tarMin, self.tarMax)
        tar_init_state = VehicleState(t=0.0,
                                      p=ParametricPose(s=s, x_tran=x_tran, e_psi=e_psi),
                                      v=BodyLinearVelocity(v_long=v_long))

        return self.genScenarioStatic(scenarioType, ego_init_state, tar_init_state)


    def randomInit(self, stateMin, stateMax):
        s = np.random.uniform(stateMin.p.s, stateMax.p.s)
        x_tran = np.random.uniform(stateMin.p.x_tran, stateMax.p.x_tran)
        e_psi = np.random.uniform(stateMin.p.e_psi, stateMax.p.e_psi)
        v_long = np.random.uniform(stateMin.v.v_long, stateMax.v.v_long)
        # print(s, x_tran, e_psi, v_long)
        return s, x_tran, e_psi, v_long
        # return 0.11891171842527565, -0.48029687041213054, 0*0.007710948256320915, 4.577211702712129

    def getLabTrack(self, ego_init_state, tar_init_state):
        return ScenarioDefinition(
            track_type='track',
            track=get_track('Monza_Track'),
            ego_init_state=ego_init_state,
            tar_init_state=tar_init_state,
            ego_obs_avoid_d=0.1,
            tar_obs_avoid_d=0.1
        )

    def genCurve(self, ego_init_state, tar_init_state):
        alpha = np.random.uniform(np.pi/5, np.pi/1.2)
        alpha = np.random.choice([-1, 1]) * alpha
        s_min = 0.5 * self.width * abs(alpha) * 1.05
        s = np.random.uniform(s_min, s_min * 2)
        return ScenarioDefinition(
            track_type='curve',
            track=CurveTrack(enter_straight_length=4,
                             curve_length=s,
                             curve_swept_angle=alpha,
                             exit_straight_length=5,
                             width=self.width,
                             slack=0.8,
                             phase_out=True,
                             ccw=False),
            ego_init_state=ego_init_state,
            tar_init_state=tar_init_state,
            ego_obs_avoid_d=0.1,
            tar_obs_avoid_d=0.1
        )

    def genStraight(self, ego_init_state, tar_init_state):
        return ScenarioDefinition(
            track_type='straight',
            track=StraightTrack(length=10, width=self.width, slack=0.8, phase_out=True),
            ego_init_state=ego_init_state,
            tar_init_state=tar_init_state,
            ego_obs_avoid_d=0.1,
            tar_obs_avoid_d=0.1
        )

    def genChicane(self, ego_init_state, tar_init_state):
        dir = np.random.choice([-1, 1])
        alpha_1 = np.random.uniform(np.pi / 7, np.pi / 3)
        alpha_1 = dir * alpha_1
        s_min = 0.5 * self.width * np.pi * abs(alpha_1) / np.pi
        s_1 = np.random.uniform(s_min, s_min * 4)

        alpha_2 = np.random.uniform(np.pi / 7, np.pi / 3)
        alpha_2 = dir * alpha_2
        s_min_2 = 0.5 * self.width * np.pi * abs(alpha_2) / np.pi
        s_2 = np.random.uniform(s_min_2, s_min_2 * 4)
        return ScenarioDefinition(
            track_type='chicane',
            track=ChicaneTrack(enter_straight_length=4,
                               curve1_length=s_1,
                               curve1_swept_angle=alpha_1,
                               mid_straight_length=0.5,
                               curve2_length=s_2,
                               curve2_swept_angle=alpha_2,
                               exit_straight_length=5,
                               width=self.width,
                               slack=0.8,
                               phase_out=True,
                               mirror=False),
            ego_init_state=ego_init_state,
            tar_init_state=tar_init_state,
            ego_obs_avoid_d=0.1,
            tar_obs_avoid_d=0.1
        )


@dataclass
class Sample():
    input: Tuple[VehicleState, VehicleState]
    output: VehicleState
    s: float


class SampleGenerator():
    '''
    Class that reads simulation results from a folder and generates samples off that for GP training. Can choose a
    function to determine whether a sample is useful or not.
    '''

    def __init__(self, abs_path, randomize=False, elect_function=None, init_all=True):
        '''
        abs path: List of absolute paths of directories containing files to be used for training
        randomize: boolean deciding whether samples should be returned in a random order or by time and file
        elect_function: decision function to choose samples
        init_all: boolean deciding whether all samples should be preloaded, can be set to False to reduce memory usage if
                        needed TODO not implemented yet!
        '''
        if elect_function is None:
            elect_function = self.useAll
        self.counter = 0
        self.abs_path = abs_path
        self.samples = []
        for ab_p in self.abs_path:
            for filename in os.listdir(ab_p):
                if filename.endswith(".pkl"):
                    dbfile = open(os.path.join(ab_p, filename), 'rb')
                    scenario_data: SimData = pickle.load(dbfile)
                    N = scenario_data.N
                    for i in range(N-1):
                        if i%3 == 0 and scenario_data.tar_preds[i] is not None:
                            ego_st = scenario_data.ego_states[i]
                            tar_st = scenario_data.tar_states[i]
                            ntar_st = scenario_data.tar_states[i + 1]
                            dtar = tar_st.copy()
                            dtar.p.s = ntar_st.p.s - tar_st.p.s
                            dtar.p.x_tran = (ntar_st.p.x_tran - tar_st.p.x_tran)
                            dtar.p.e_psi = ntar_st.p.e_psi - tar_st.p.e_psi
                            dtar.v.v_long = ntar_st.v.v_long - tar_st.v.v_long
                            dtar.w.w_psi = ntar_st.w.w_psi - tar_st.w.w_psi
                            if elect_function(ego_st, tar_st):
                                self.samples.append(Sample((ego_st, tar_st), dtar, tar_st.lookahead.curvature[0]))
                    dbfile.close()
        print('Generated Dataset with', len(self.samples), 'samples!')
        if randomize:
            random.shuffle(self.samples)

    def reset(self, randomize=False):
        if randomize:
            random.shuffle(self.samples)
        self.counter = 0

    def getNumSamples(self):
        return len(self.samples)

    def nextSample(self):
        self.counter += 1
        if self.counter >= len(self.samples):
            print('All samples returned. To reset, call SampleGenerator.reset(randomize)')
            return None
        else:
            return self.samples[self.counter - 1]

    def plotStatistics(self, param):
        data_list = []
        if param == 'c':
            for i in self.samples:
                data_list.append(i.input[1].lookahead.curvature[0])
        fig, axs = plt.subplots(1, 1)
        axs.hist(data_list, bins=50)
        plt.show()

    def even_augment(self, param, maxQ):
        '''
        Augments the dataset to even out distribution of samples by param
        Input:
            param: param to even out by
            maxQ: max factor of frequency of samples
        '''
        data_list = []
        n_bins = 10
        if param == 's':
            for i in self.samples:
                data_list.append(i.s)
        hist, bins = np.histogram(data_list, bins=n_bins)
        maxCount = np.max(hist)  # highest frequency bin
        samples_binned = []
        samples = [k for k in self.samples if k.s < bins[0]]
        samples_binned.append(samples)
        for i in range(n_bins-2):
            samples = [k for k in self.samples if bins[i + 1] > k.s >= bins[i]]
            samples_binned.append(samples)
        samples = [k for k in self.samples if k.s >= bins[-1]]
        samples_binned.append(samples)
        for i in samples_binned:
            if len(i) > 0:
                fac = int(round(maxCount*maxQ/(len(i))))
                print(maxCount*maxQ/ (len(i)), fac, maxCount, len(i))
                for j in range(fac-1):
                    self.samples.extend(i)
        self.plotStatistics('s')
        self.reset(randomize=True)


    def useAll(self, ego_state, tar_state):
        return True

def derive_lateral_long_error_from_true_traj(sim_data : EvalData):
    lateral_error = list()
    longitudinal_error = list()
    track = sim_data.scenario_def.track
    samps = 0
    for timeStep in range(len(sim_data.tar_states)):
        pred = sim_data.tar_gp_pred[timeStep]  # (VehiclePrediction) at current timestep, what is GP prediction
        if pred is not None:
            N = len(pred.s) if pred.s else len(pred.x)
            if N + timeStep - 1 < len(sim_data.tar_states):
                samps += 1
                for i in range(1, N):
                    tar_st = sim_data.tar_states[timeStep + i - 1]  # (VehicleState) current target state from true traveled trajectory
                    if pred.s:
                        current_x, current_y, current_psi = track.local_to_global(
                            (pred.s[i], pred.x_tran[i], pred.e_psi[i]))
                        track.local_to_global_typed(tar_st)
                    else:
                        current_x, current_y, current_psi = pred.x[i], pred.y[i], pred.psi[i]

                    dx = tar_st.x.x - current_x
                    dy = tar_st.x.y - current_y

                    longitudinal = dx * np.cos(current_psi) + dy * np.sin(current_psi)
                    lateral = -dx * np.sin(current_psi) + dy * np.cos(current_psi)
                    longitudinal_error.append(longitudinal)
                    lateral_error.append(lateral)

    return lateral_error, longitudinal_error

def derive_lateral_long_error_from_MPC_preds(sim_data : EvalData):
    lateral_error = list()
    longitudinal_error = list()
    track = sim_data.scenario_def.track
    samps = 0
    for timeStep in range(len(sim_data.tar_states)):
        pred_gp = sim_data.tar_gp_pred[timeStep]  # (VehiclePrediction) at current timestep, what is GP prediction
        pred_mpc = sim_data.tar_preds[timeStep]  # (VehiclePrediction) at current timestep, what is MPCC prediction
        if pred_gp is not None and pred_mpc is not None:
            N = len(pred_gp.s) if pred_gp.s else len(pred_gp.x)
            if N + timeStep - 1 < len(sim_data.tar_states):
                samps += 1
                for i in range(1, N):
                    if pred_gp.s:
                        current_x, current_y, current_psi = track.local_to_global(
                            (pred_gp.s[i], pred_gp.x_tran[i], pred_gp.e_psi[i]))
                    else:
                        current_x, current_y, current_psi = pred_gp.x[i], pred_gp.y[i], pred_gp.psi[i]

                    dx = pred_mpc.x[i] - current_x
                    dy = pred_mpc.y[i] - current_y

                    longitudinal = dx * np.cos(current_psi) + dy * np.sin(current_psi)
                    lateral = -dx * np.sin(current_psi) + dy * np.cos(current_psi)
                    longitudinal_error.append(longitudinal)
                    lateral_error.append(lateral)

    return lateral_error, longitudinal_error

def evaluateGPNew(sim_data: List[EvalData]):
    import seaborn as sns

    lateral_error_gp, longitudinal_error_gp = derive_lateral_long_error_from_true_traj(sim_data[0])
    lateral_error_cv, longitudinal_error_cv = derive_lateral_long_error_from_true_traj(sim_data[1])
    lateral_error_ca, longitudinal_error_ca = derive_lateral_long_error_from_true_traj(sim_data[2])
    sim_data[2].tar_gp_pred = sim_data[2].tar_preds
    lateral_error_mpc, longitudinal_error_mpc = derive_lateral_long_error_from_true_traj(sim_data[2])

    plt.subplot(1, 2, 1)
    plt.xlabel("Lateral error (m)")
    sns.distplot(lateral_error_gp, color="r")
    sns.distplot(lateral_error_cv, color="b")
    sns.distplot(lateral_error_ca, color="g")
    sns.distplot(lateral_error_mpc, color="y")
    plt.legend(["GP", "CV", "CA", "MPC"])
    plt.ylim((0, 20))

    plt.subplot(1, 2, 2)
    plt.xlabel("Longitudinal error (m)")
    sns.distplot(longitudinal_error_gp, color="r")
    sns.distplot(longitudinal_error_cv, color="b")
    sns.distplot(longitudinal_error_ca, color="g")
    sns.distplot(longitudinal_error_mpc, color="y")
    plt.ylim((0, 20))

    plt.suptitle("Comparison of prediction modules \nagainst ground truth trajectory")
    plt.show()




def evaluateGP(sim_data: SimData, post_gp_eval):
    devs = 0
    devx_tran = 0
    devs_p = 0
    devx_tran_p = 0
    maxs = 0
    max_tran = 0
    maxs_p = 0
    max_tran_p = 0
    samps = 0
    for timeStep in range(len(sim_data.tar_states)):
        pred = sim_data.tar_gp_pred[timeStep]
        if post_gp_eval:
            pred_p = sim_data.tar_gp_pred_post[timeStep]
        s_temp = 0
        tran_temp = 0
        s_temp_p = 0
        tran_temp_p = 0
        if pred is not None:
            N = len(pred.s)
            if N + timeStep - 1 < len(sim_data.tar_states):
                samps += 1
                for i in range(1, N):
                    tar_st = sim_data.tar_states[timeStep + i - 1]
                    ds = (pred.s[i] - tar_st.p.s) ** 2
                    dtran = (pred.x_tran[i] - tar_st.p.x_tran) ** 2
                    s_temp += ds / N
                    tran_temp += dtran / N
                    if ds > maxs:
                        maxs = ds
                    if dtran > max_tran:
                        max_tran = dtran
                    if post_gp_eval:
                        ds_p = (pred_p.s[i] - tar_st.p.s) ** 2
                        dtran_p = (pred_p.x_tran[i] - tar_st.p.x_tran) ** 2
                        s_temp_p +=  ds_p / N
                        tran_temp_p +=  dtran_p/ N
                        if ds_p > maxs_p:
                            maxs_p = ds_p
                        if dtran_p > max_tran_p:
                            max_tran_p = dtran_p
                '''s_temp += (pred.s[i] - tar_st.p.s)**2/(N*(max((ego_pred.s[i] - ego_st.p.s)**2, 0.001)))
                    tran_temp += (pred.x_tran[i] - tar_st.p.x_tran)**2/(N*(max((ego_pred.x_tran[i] - ego_st.p.x_tran)**2, 0.001)))'''
                devs += s_temp
                devx_tran += tran_temp
                if post_gp_eval:
                    devs_p += s_temp_p
                    devx_tran_p += tran_temp_p
    devs /= samps
    devx_tran /= samps
    devs_p /= samps
    devx_tran_p /= samps
    if post_gp_eval:
        print('Real sim predictions')
    print('Avg s pred squared error: ', devs, 'max s squared error ', maxs)
    print('Avg x_tran squared error: ', devx_tran,
          ' max x_tran squared error ', max_tran)
    if post_gp_eval:
        print('Post sim predictions')
        print('Avg s pred accuracy: ', devs_p, 'max s squared error ', maxs_p)
        print('Avg x_tran pred accuracy: ', devx_tran_p,
              ' max x_tran squared error ', max_tran_p)


def post_gp(sim_data: EvalData, gp):
    """
    Gets GP predictions based on true ego predictions, eval purposes only
    """
    track_obj = sim_data.scenario_def.track
    sim_data.tar_gp_pred_post = [None]*len(sim_data.tar_gp_pred)
    for timeStep in range(len(sim_data.tar_states)):
        pred = sim_data.tar_gp_pred[timeStep]
        ego_pred = sim_data.ego_preds[timeStep] # TODO check if first entry contains current state
        if pred is not None and ego_pred is not None:
            N = len(pred.s)
            ego_pred_real = ego_pred.copy()
            if N + timeStep - 1 < len(sim_data.tar_states):
                for i in range(1,len(ego_pred_real.s)):
                    c_state = sim_data.ego_states[timeStep - 1 + i]
                    ego_pred_real.s[i] = c_state.p.s
                    ego_pred_real.x_tran[i] = c_state.p.x_tran
                    ego_pred_real.v_long[i] = c_state.v.v_long
                    ego_pred_real.e_psi[i] = c_state.p.e_psi
                pred = gp.get_true_prediction_par(sim_data.ego_states[timeStep-1], sim_data.tar_states[timeStep-1], ego_pred_real, track_obj, M=50)
                sim_data.tar_gp_pred_post[timeStep] = pred


def smoothPlotResults(sim_data: SimData, speedup=1, fps=60, start_t=0, close_loop=False):
    def on_release(key):
        if key == pynput.keyboard.Key.enter:
            # Stop listener
            return False
    def on_press(key):
        if key == pynput.keyboard.Key.enter:
            # Stop listener
            return False

    print('Plotting simulation of length t=', sim_data.ego_states[-1].t, " with ", speedup, "x speed")
    plot_conf = GlobalPlotConfigs(buffer_length=50, draw_period=1 / (2 * fps), update_period=1 / (2 * fps),
                                  close_loop=close_loop)
    ego_v_plot_conf = VehiclePlotConfigs('ego', vehicle_draw_L=0.26, vehicle_draw_W=.173, show_sim=True, simulated=True,
                                         show_est=False, show_ecu=True, show_pred=True, show_traces=True, show_full_traj=True,
                                         state_list = sim_data.ego_states, color='g')
    tar_v_plot_conf = VehiclePlotConfigs('tar', vehicle_draw_L=0.26, vehicle_draw_W=.173, show_sim=True, simulated=True,
                                         show_est=False, show_ecu=True, show_pred=True,
                                         show_traces=False, show_full_traj=True, state_list=sim_data.tar_states)
    fig = BarcFigure(0, plot_conf)
    fig.track = sim_data.scenario_def.track
    fig.add_vehicle(ego_v_plot_conf)
    fig.add_vehicle(tar_v_plot_conf)
    if hasattr(sim_data, "tar_gp_pred") and sim_data.tar_gp_pred is not None:
        gp_tar_v_plot_conf = VehiclePlotConfigs('gp_tar', vehicle_draw_L=0.26, vehicle_draw_W=.173, show_sim=True,
                                             simulated=True,
                                             show_est=False, show_ecu=True, show_pred=True,
                                             show_cov=True, # Enable covariance plotting
                                             show_traces=False, show_full_vehicle_bodies=True, color='r')
        fig.add_vehicle(gp_tar_v_plot_conf)
        # prepare local to global plot data. Barc plotter can't handle that because track model will be changed
        '''for j in range(len(sim_data.tar_gp_pred)):
            if sim_data.tar_gp_pred[j] is not None:
                sim_data.tar_gp_pred[j].x = []
                sim_data.tar_gp_pred[j].y = []
                sim_data.tar_gp_pred[j].psi = []
                for i in range(len(sim_data.tar_gp_pred[j].s)):
                    (x, y, psi) = fig.track.local_to_global([sim_data.tar_gp_pred[j].s[i], sim_data.tar_gp_pred[j].x_tran[i], sim_data.tar_gp_pred[j].e_psi[i]])
                    sim_data.tar_gp_pred[j].x.append(x)
                    sim_data.tar_gp_pred[j].y.append(y)
                    sim_data.tar_gp_pred[j].psi.append(psi)'''
    fig.track.remove_phase_out()
    p = mp.Process(target=fig.run_plotter)
    p.start()
    i = 0
    i2 = 0
    j = 0
    k = 0
    start_id = 0
    if start_t != 0:
        start_id = next(k for k,x in enumerate(sim_data.ego_states) if x.t >= start_t)
    i+= start_id
    ego_state = sim_data.ego_states[i]
    ego_interp = interp_data(sim_data.ego_states[start_id:], speedup * 1 / fps)
    tar_interp = interp_data(sim_data.tar_states[start_id:], speedup * 1 / fps)
    tar_state = sim_data.tar_states[i]
    not_done = True
    fig.update('ego', ego_state)
    fig.update('tar', tar_state)
    print("Press 'Enter' to play simulation")

    with pynput.keyboard.Listener(on_press=on_press,
            on_release=on_release) as listener:
        listener.join()
    start_time = time.time()
    itemp = 0
    while not_done:
        dtime = time.time() - start_time
        j += 1
        ego_state.x.x = ego_interp['x'][k]
        ego_state.x.y = ego_interp['y'][k]
        ego_state.e.psi = ego_interp['psi'][k]
        ego_state.p.s = sim_data.ego_states[i].p.s
        tar_state.x.x = tar_interp['x'][k]
        tar_state.x.y = tar_interp['y'][k]
        tar_state.e.psi = tar_interp['psi'][k]
        if dtime < 1 / fps:
            time.sleep(1 / fps - dtime)
        fig.update('ego', sim_data=ego_state)
        fig.update('tar', tar_state)
        k += 1
        if sim_data.ego_preds[i] is not None and sim_data.ego_preds[i].x is not None:
            fig.update_vehicle_prediction('ego', sim_data.ego_preds[i])
            '''act = sim_data.ego_states[i].u
            act.t = sim_data.ego_states[i].t
            fig.update('ego', ecu_data=sim_data.ego_states[i].u)'''
        if sim_data.tar_preds[i] is not None and sim_data.tar_preds[i].x is not None:
            fig.update_vehicle_prediction('tar', sim_data.tar_preds[i])
            pass

        # GP predictions
        if hasattr(sim_data, "tar_gp_pred") and sim_data.tar_gp_pred is not None and sim_data.tar_gp_pred[i] is not None:
            fig.update('gp_tar', sim_data.tar_states[i])
            fig.update_vehicle_prediction('gp_tar', sim_data.tar_gp_pred[i])
            '''if i < len(sim_data.tar_states)-1 and not i == itemp:
                print('True u_a', sim_data.tar_states[i+1].u.u_a, 'u_s', sim_data.tar_states[i+1].u.u_steer)
                print('Pred u_a', sim_data.tar_gp_pred[i].u.u_a, 'u_s', sim_data.tar_gp_pred[i].u.u_steer)
                itemp = i'''
        while sim_data.ego_states[i].t <= speedup * j * 1 / fps + start_t:
            i += 1
            if i >= len(sim_data.ego_states):
                break
        if i < len(sim_data.ego_states) and i < len(sim_data.tar_states) and k < len(ego_interp['x']):
            pass
        else:
            not_done = False
        start_time = time.time()
    p.join()
    p.close()


def interp_data(state_list, dt, kind='quadratic'):
    x = []
    y = []
    psi = []
    t = []
    t_end = state_list[-1].t - state_list[0].t
    for st in state_list:
        x.append(st.x.x)
        y.append(st.x.y)
        psi_ = st.e.psi
        while len(psi) > 0 and abs(psi[-1] - psi_) > np.pi / 2:
            if psi[-1] - psi_ > np.pi / 2:
                psi_ = psi_ + np.pi
            else:
                psi_ = psi_ - np.pi
        psi.append(psi_)
        t.append(st.t- state_list[0].t)
    t = np.array(t)
    x = np.array(x)
    y = np.array(y)
    psi = np.array(psi)
    interpStates = dict()
    x_interp = scipy.interpolate.interp1d(t, x, kind=kind)
    y_interp = scipy.interpolate.interp1d(t, y, kind=kind)
    psi_interp = scipy.interpolate.interp1d(t, psi, kind=kind)
    interpStates['x'] = [float(x_interp(dt * i)) for i in range(round(t_end / dt))]
    interpStates['y'] = [float(y_interp(dt * i)) for i in range(round(t_end / dt))]
    interpStates['psi'] = [float(psi_interp(dt * i)) for i in range(round(t_end / dt))]
    interpStates['t'] = [dt * i for i in range(round(t_end / dt))]
    return interpStates


def interp_state(state1, state2, t):
    state = state1.copy()
    dt0 = t - state1.t
    dt = state2.t - state1.t
    state.p.s = (state2.p.s - state1.p.s) / dt * dt0 + state1.p.s
    state.p.x_tran = (state2.p.x_tran - state1.p.x_tran) / dt * dt0 + state1.p.x_tran
    state.x.x = (state2.x.x - state1.x.x) / dt * dt0 + state1.x.x
    state.x.y = (state2.x.y - state1.x.y) / dt * dt0 + state1.x.y
    state.e.psi = (state2.e.psi - state1.e.psi) / dt * dt0 + state1.e.psi
    return state


##############################
# Example scenario Definitions
##############################

'''
Scenario description:
- straight track
- ego to the left and behind target
- ego same speed as target
'''