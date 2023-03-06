#!/usr/bin/env python3

from tqdm import tqdm
import seaborn as sns
from typing import List
import cProfile, pstats, io
from pstats import SortKey
import tikzplotlib
import matplotlib.pyplot as plt

from barcgp.common.utils.file_utils import *
from barcgp.common.utils.scenario_utils import EvalData, PostprocessData

from barcgp.h2h_configs import *

total_runs = 100
track_width = 1.5
blocking_threshold = 0.2  # Percentage of track x_tran movement to consider for blocking

policy_name = 'aggressive_blocking'
policy_dir = os.path.join(eval_dir, policy_name)
scen_dir = os.path.join(policy_dir, 'track')

names = sorted(os.listdir(scen_dir), key=str)
print(names)
colors = {"GP": "r", "NLMPC": "b", "CAV": "g", "CV": "m", "MPCC": "k", "STSP": "y"}
find_color = lambda pred: [val for key, val in colors.items() if key in pred].__getitem__(0)


@dataclass
class Metrics:
    crash: bool = field(default=False)
    ego_crash: bool = field(default=False)
    tv_crash: bool = field(default=False)
    time_crash: float = field(default=0)
    ego_win: bool = field(default=False)
    left_track: bool = field(default=False)
    left_track_tv: bool = field(default=False)
    delta_s: float = field(default=0)
    time_overtake: float = field(default=-1)
    s_overtake: float = field(default=-1)
    xy_overtake: np.array = field(default=None)
    xy_crash: np.array = field(default=None)
    overtake_attempts_l: int = field(default=0)
    overtake_attempts_r: int = field(default=0)
    d_s_cont: np.array = field(default=None)
    col_d: np.array = field(default=None)
    init_ds: float = field(default=0)
    lateral_error: np.array = field(default=np.array([]))
    longitudinal_error: np.array = field(default=np.array([]))
    lateral_rmse: float = field(default=0.0)
    longitudinal_rmse: float = field(default=0.0)
    u_a: np.array = field(default=np.array([]))
    u_a_min: float = field(default=0)
    u_s: np.array = field(default=np.array([]))
    tv_inf: bool = field(default=None)
    ego_inf: bool = field(default=None)


@dataclass
class Vehicle:
    l: float = field(default=None)
    w: float = field(default=None)
    x: float = field(default=None)
    y: float = field(default=None)
    phi: float = field(default=None)


def collision_check(e1: Vehicle, e2: Vehicle):
    posx = e1.x
    posy = e1.y
    x_ob = e2.x
    y_ob = e2.y
    l_ob = e2.l
    w_ob = e2.w
    r = e1.w / 2
    dl = e1.l * 0.9 / 3
    s_e = np.sin(e1.phi)
    c_e = np.cos(e1.phi)
    s = np.sin(e2.phi)
    c = np.cos(e2.phi)
    dx1 = x_ob - (posx - 3 * dl * c_e / 2)
    dx2 = x_ob - (posx - dl * c_e / 2)
    dx3 = x_ob - (posx + dl * c_e / 2)
    dx4 = x_ob - (posx + 3 * dl * c_e / 2)
    dy1 = y_ob - (posy - 3 * dl * s_e / 2)
    dy2 = y_ob - (posy - dl * s_e / 2)
    dy3 = y_ob - (posy + dl * s_e / 2)
    dy4 = y_ob - (posy + 3 * dl * s_e / 2)
    a = (l_ob / np.sqrt(2) + r) * 1
    b = (w_ob / np.sqrt(2) + r) * 1

    i1 = (c * dx1 - s * dy1) ** 2 * 1 / a ** 2 + (s * dx1 + c * dy1) ** 2 * 1 / b ** 2
    i2 = (c * dx2 - s * dy2) ** 2 * 1 / a ** 2 + (s * dx2 + c * dy2) ** 2 * 1 / b ** 2
    i3 = (c * dx3 - s * dy3) ** 2 * 1 / a ** 2 + (s * dx3 + c * dy3) ** 2 * 1 / b ** 2
    i4 = (c * dx4 - s * dy4) ** 2 * 1 / a ** 2 + (s * dx4 + c * dy4) ** 2 * 1 / b ** 2
    return i1 > 1 and i2 > 1 and i3 > 1 and i4 > 1, min([i1, i2, i3, i4])


class CollisionChecker():
    def __init__(self, ego_win):
        self.t_col = 0
        self.col_d = []
        self.col = False
        self.s = []
        self.ego_win = ego_win
        self.collision_xy = None
        self.collision_y = 0
        self.tar_leading=True
        self.ego_leading_steps=0
        self.col_ego = False
        self.col_tv = False

    def next(self, e: VehicleState, t: VehicleState, e_p, t_p):
        if e.p.s > t.p.s:
            self.ego_leading_steps+=1
            if self.ego_leading_steps > 1:
                self.tar_leading=False
        e1 = Vehicle(l=ego_L, w=ego_W, x=e.x.x, y=e.x.y, phi=e.e.psi)
        e2 = Vehicle(l=tar_L, w=tar_W, x=t.x.x, y=t.x.y, phi=t.e.psi)
        res, d = collision_check(e1, e2)
        self.col_d.append(d)
        if self.ego_win:
            self.s.append(e.p.s)
        else:
            self.s.append(t.p.s)
        if not self.col and not res:
            self.t_col = e.t
            self.col = True
            if self.tar_leading:
                self.col_ego = True
            else:
                self.col_tv = True
            self.collision_xy = [e.x.x, e.x.y]

    def get_results(self):
        s = np.array(self.s)
        s = s / s[-1] * 100
        i = np.arange(100)
        interpolated = np.interp(i, s, self.col_d)
        return (self.col, self.t_col, interpolated, self.collision_xy, self.col_ego, self.col_tv)


class InterpolatedLead():
    def __init__(self, ego_win):
        self.ds = []
        self.s_p = []
        self.ds_percent = []
        self.ego_win = ego_win

    def next(self, e: VehicleState, t: VehicleState, e_p, t_p):
        self.ds.append(e.p.s - t.p.s)
        if self.ego_win:
            self.s_p.append(e.p.s)
        else:
            self.s_p.append(t.p.s)

    def get_results(self):
        s_p = np.array(self.s_p)
        s_p = s_p / s_p[-1] * 100
        i = np.arange(100)
        interpolated = np.interp(i, s_p, self.ds)
        return interpolated


class InterpolatedActuation():
    def __init__(self):
        self.u_a_min = np.inf
        self.u_a = []
        self.u_s = []
        self.s_p = []

    def next(self, e: VehicleState, t, e_p, t_p):
        if e.u.u_a < self.u_a_min:
            self.u_a_min = e.u.u_a
        self.u_a.append(e.u.u_a)
        self.u_s.append(e.u.u_steer)
        self.s_p.append(e.p.s)

    def get_results(self):
        s_p = np.array(self.s_p)
        s_p = s_p / s_p[-1] * 100
        i = np.arange(100)
        u_a = np.interp(i, s_p, self.u_a)
        u_s = np.interp(i, s_p, self.u_s)
        return [u_a, u_s, self.u_a_min]


class LeftTrack():
    def __init__(self, track_width):
        self.left = False
        self.left_tv = False
        self.track_wdth = track_width

    def next(self, e: VehicleState, t, e_p, t_p):
        if not self.left:
            if abs(e.p.x_tran) > self.track_wdth / 2 + ego_W:
                self.left = True
        if not self.left_tv:
            if abs(t.p.x_tran) > self.track_wdth / 2 + tar_W:
                self.left_tv = True

    def get_results(self):
        return self.left, self.left_tv


class Overtake():
    def __init__(self):
        self.tar_leading = True
        self.overtake_t = -1
        self.overtake_s = -1
        self.overtake_xy = [-1, -1]
        self.o_a_l = 0
        self.o_a_r = 0

    def next(self, e: VehicleState, t: VehicleState, e_p: VehiclePrediction, t_p: VehiclePrediction):
        if e.p.s > t.p.s and self.tar_leading:
            self.tar_leading = False
            self.overtake_t = e.t
            self.overtake_s = e.p.s
            self.overtake_xy = [e.x.x, e.x.y]
        elif e.p.s < t.p.s:
            if e_p is not None and e_p.x_tran is not None:
                if np.mean(e_p.x_tran) > t.p.x_tran:
                    self.o_a_r += 1
                else:
                    self.o_a_l += 1

    def get_results(self):
        all = max(1, self.o_a_l + self.o_a_r)
        o_a_l = self.o_a_l / all
        o_a_r = self.o_a_r / all
        return self.overtake_t, self.overtake_s, self.overtake_xy, o_a_l, o_a_r


def derive_lateral_long_error_from_true_traj(sim_data : EvalData, check_only_blocking=False):
    """
    @param sim_data: Input evaluation data where we are comparing against `tar_states` (true trajectory)
    @return:
    lateral_error (list of l_1 errors)
    longitudinal_error (list of l_1 errors)
    """
    lateral_error = []
    longitudinal_error = []
    track = sim_data.scenario_def.track
    samps = 0
    for timeStep in range(len(sim_data.tar_states)-1):
        pred = sim_data.tar_gp_pred[timeStep]  # (VehiclePrediction) at current timestep, what is GP prediction
        if pred is not None and (pred.x is not None or pred.s is not None):
            N = len(pred.s) if pred.s else len(pred.x)
            if N + timeStep - 1 < len(sim_data.tar_states):
                samps += 1
                for i in range(1, N):
                    tar_st = sim_data.tar_states[timeStep + i]  # (VehicleState) current target state from true traveled trajectory

                    if check_only_blocking:
                        xt = tar_st.p.x_tran
                        x_ref = np.sign(xt) * min(track.half_width, abs(float(xt)))

                        if pred.s is None and pred.x is not None:
                            f = track.global_to_local((pred.x[i], pred.y[i], pred.psi[i]))
                            if f is not None:
                                s, x_tran, e_psi = f
                            else:
                                return np.array([]), np.array([])
                            if s <= tar_st.p.s and np.fabs(x_ref) > blocking_threshold * track.track_width:
                                longitudinal = s - tar_st.p.s
                                lateral = x_tran - tar_st.p.x_tran
                                longitudinal_error.append(longitudinal)
                                lateral_error.append(lateral)
                        else:
                            if pred.s[i] <= tar_st.p.s and np.fabs(x_ref) > blocking_threshold * track.track_width:
                                longitudinal = pred.s[i] - tar_st.p.s
                                lateral = pred.x_tran[i] - tar_st.p.x_tran
                                longitudinal_error.append(longitudinal)
                                lateral_error.append(lateral)
                    else:
                        if not pred.s:
                            dx = tar_st.x.x - pred.x[i]
                            dy = tar_st.x.y - pred.y[i]
                            angle = sim_data.scenario_def.track.local_to_global((tar_st.p.s, 0, 0))[2]
                            longitudinal = dx * np.cos(angle) + dy * np.sin(angle)
                            lateral = -dx * np.sin(angle) + dy * np.cos(angle)
                        else:
                            longitudinal = pred.s[i] - tar_st.p.s
                            lateral = pred.x_tran[i] - tar_st.p.x_tran
                        longitudinal_error.append(longitudinal)
                        lateral_error.append(lateral)

    return np.array(lateral_error), np.array(longitudinal_error)


def get_metrics(scen_data: EvalData):
    # TODO: This iterates over all times multiple times! Fix this
    metrics = Metrics()
    ego_win = scen_data.ego_states[-1].p.s > scen_data.tar_states[-1].p.s
    Col = CollisionChecker(ego_win)
    Lead = InterpolatedLead(ego_win)
    Act = InterpolatedActuation()
    L_Track = LeftTrack(scen_data.scenario_def.track.track_width)
    OT = Overtake()
    for (e, t, ep, tp) in zip(scen_data.ego_states, scen_data.tar_states, scen_data.ego_preds, scen_data.tar_preds):
        Col.next(e, t, ep, tp)
        Lead.next(e, t, ep, tp)
        L_Track.next(e, t, ep, tp)
        Act.next(e, t, ep, tp)
        OT.next(e, t, ep, tp)

    cr, t, inter, col_xy, metrics.ego_crash, metrics.tv_crash = Col.get_results()
    metrics.crash = cr
    if cr:
        metrics.crash_t = t
    metrics.delta_s = scen_data.ego_states[-1].p.s - scen_data.tar_states[-1].p.s
    o_t, o_s, o_xy, o_l, o_r = OT.get_results()
    metrics.time_overtake = o_t
    if hasattr(scen_data, 'tv_infeasible'):
        metrics.ego_inf, metrics.tv_inf = scen_data.ego_infeasible, scen_data.tv_infeasible
    while o_s > scen_data.scenario_def.track.track_length:
        o_s -= scen_data.scenario_def.track.track_length
    metrics.s_overtake = o_s
    metrics.xy_overtake = o_xy
    metrics.xy_crash = col_xy
    metrics.overtake_attempts_l = o_l
    metrics.overtake_attempts_r = o_r
    metrics.left_track, metrics.left_track_tv = L_Track.get_results()
    metrics.ego_win = ego_win and not cr and not metrics.left_track
    metrics.d_s_cont = Lead.get_results()
    metrics.col_d = inter
    metrics.init_ds = scen_data.ego_states[0].p.s - scen_data.tar_states[0].p.s
    metrics.lateral_error, metrics.longitudinal_error = derive_lateral_long_error_from_true_traj(scen_data, check_only_blocking=True)
    u = Act.get_results()
    metrics.lateral_rmse = np.sqrt(np.mean(metrics.lateral_error ** 2))
    metrics.longitudinal_rmse = np.sqrt(np.mean(metrics.longitudinal_error ** 2))
    metrics.u_a = u[0]
    metrics.u_s = u[1]
    metrics.u_a_min = u[2]
    return metrics

def parse_metrics(metrics: Metrics, data: PostprocessData, i):
    # Counts
    data.num_wins += metrics.ego_win
    data.num_overtakes += metrics.ego_win  # TODO Replace this
    data.num_ego_inf += metrics.ego_inf
    data.num_tv_inf += metrics.tv_inf
    data.num_crashes += metrics.crash
    data.num_left_track += metrics.left_track
    # Averages
    data.avg_delta_s += metrics.delta_s/data.N
    data.avg_a += np.average(metrics.u_a)/data.N
    data.avg_min_a += metrics.u_a_min/data.N
    data.avg_abs_steer += np.average(np.abs(metrics.u_s))/data.N
    if metrics.ego_win:
        data.overtake_s.append(metrics.s_overtake)
        xy = metrics.xy_overtake
        data.overtake_x.append(xy[0])
        data.overtake_y.append(xy[1])
        data.win_ids.append(str(i) + '.pkl')
        data.overtake_ids.append(str(i) + '.pkl')
    if metrics.ego_inf:
        data.ego_infesible_ids.append(str(i) + '.pkl')
    if metrics.tv_inf:
        data.tv_infesible_ids.append(str(i) + '.pkl')
    if metrics.crash:
        data.crash_ids.append(str(i) + '.pkl')
        data.crash_x.append(metrics.xy_crash[0])
        data.crash_y.append(metrics.xy_crash[1])
        if metrics.ego_crash:
            data.crash_ids_ego.append(str(i) + '.pkl')
        else:
            data.crash_ids_tv.append(str(i) + '.pkl')
    if metrics.left_track:
        data.left_track_ids.append(str(i) + '.pkl')
    if not metrics.left_track_tv:
        data.lateral_errors.extend(metrics.lateral_error)
        data.longitudinal_errors.extend(metrics.longitudinal_error)


def main(args=None):
    print(f"Evaluating data for policy: {policy_name} in {scen_dir}")
    n = len(names)
    # Initialize processed data container
    processed_data = {}
    for i in names:
        processed_data[i] = PostprocessData()
        processed_data[i].name = i
        processed_data[i].setup_id = policy_name
        processed_data[i].N = 100

    # Temporary variables
    counter = [0] * n
    counter_nc = [0] * n
    vals = [0] * n
    wins = [0] * n
    win_ids = [None] * n
    tv_inf = [0] * n
    ego_inf = [0] * n
    fair_counter = 0
    fair_vals = [0] * n
    crashes = [0] * n
    o_l = [0] * n
    o_r = [0] * n
    o_s = []
    o_xy = []
    for i in range(n):
        o_s.append([])
        o_xy.append([])
    l_t = [0] * n
    d_s_cont = [np.zeros((100,))]
    col_d_cont = [np.zeros((100,))]
    u_a = [np.zeros((100,))]
    u_s = [np.zeros((100,))]
    lat_rmse = [0] * n
    long_rmse = [0] * n
    lateral_errors, longitudinal_errors = dict(), dict()
    for i in range(n - 1):
        d_s_cont.append(np.zeros((100,)))
        col_d_cont.append(np.zeros((100,)))
        u_a.append(np.zeros((100,)))
        u_s.append(np.zeros((100,)))
    pr = cProfile.Profile()
    pr.enable()
    for i in tqdm(range(total_runs)):
        scores = [None] * n
        exists = True
        for a in names:
            name = os.path.join(os.path.join(scen_dir, a), str(i) + '.pkl')
            if not os.path.exists(name):
                exists = False
                break
        if exists:
            for id, a in enumerate(names):
                if id not in lateral_errors:
                    lateral_errors[id] = []
                if id not in longitudinal_errors:
                    longitudinal_errors[id] = []

                name = os.path.join(os.path.join(scen_dir, a), str(i) + '.pkl')
                if os.path.exists(name):
                    dbfile = open(name, 'rb')
                    scenario_data: EvalData = pickle.load(dbfile)
                    if processed_data[a].track is None:
                        processed_data[a].track = scenario_data.scenario_def.track
                    metrics = get_metrics(scenario_data)
                    parse_metrics(metrics, processed_data[a], i)

                    scores[id] = metrics.delta_s
                    counter[id] += 1
                    vals[id] += scores[id]

                    if not metrics.crash:  # and not metrics.left_track:
                        counter_nc[id] += 1
                        d_s_cont[id] += metrics.d_s_cont
                    o_l[id] += metrics.overtake_attempts_l
                    o_r[id] += metrics.overtake_attempts_r
                    if metrics.ego_win:
                        o_s[id].append(metrics.s_overtake)
                        o_xy[id].append(metrics.xy_overtake)
                    l_t[id] += metrics.left_track
                    col_d_cont[id] += metrics.col_d
                    u_s[id] += metrics.u_s
                    u_a[id] += metrics.u_a
                    if not metrics.crash and not metrics.left_track and not metrics.left_track_tv:
                        lateral_errors[id].extend(metrics.lateral_error)
                        longitudinal_errors[id].extend(metrics.longitudinal_error)
                    lat_rmse[id] += metrics.lateral_rmse
                    long_rmse[id] += metrics.longitudinal_rmse

            if all(s is not None for s in scores):
                fair_counter += 1
                for i in range(n):
                    fair_vals[i] += scores[i]
    for a in names:
        processed_data[a].post()
    post_path = os.path.join(policy_dir, policy_name + '.pkl')
    pickle_write(processed_data, post_path)
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    print("============== AVG =============")
    avg_delta_s_print = ["AVG Delta S - "]
    w = ["Wins - "]
    wr = ["Win Rate - "]
    wr_nc = ["Win Rate nc - "]
    sr = ["Succesful Runs - "]
    ct = ["Crashes total - "]
    ctr = ["Crashes total rate - "]
    ol = ["Overtake attempts Left - "]
    or_ = ["Overtake attempts Right - "]
    os_ = ["AVG Overtake s - "]
    lt_ = ["Left track rate - "]
    us = ["AVG u_s - "]
    ua = ["AVG u_a - "]
    lat_rmse_ = ["Lateral RMSE - "]
    long_rmse_ = ["Longitudinal RMSE - "]
    ego_infeasible = ["Ego Infeas - "]
    tv_infeasible = ["TV Infeas - "]
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    for i in range(len(names)):
        counter[i] = max(1, counter[i])
        processed_data[names[i]].num_crashes = counter[i] - counter_nc[i]
        counter_nc[i] = max(1, counter_nc[i])
        avg_delta_s_print.extend([names[i] + ": ", "{:.3f}".format(vals[i] / counter[i])])
        processed_data[names[i]].avg_delta_s = vals[i] / counter[i]
        w.extend([names[i] + ": ", wins[i]])
        wr.extend([names[i] + ": ", "{:.3f}".format(wins[i] / counter[i])])
        wr_nc.extend([names[i] + ": ", "{:.3f}".format(wins[i] / counter_nc[i])])
        sr.extend([names[i] + ": ", counter[i]])
        ct.extend([names[i] + ": ", crashes[i]])
        ctr.extend([names[i] + ": ", "{:.3f}".format(crashes[i] / counter[i])])
        ol.extend([names[i] + ": ", "{:.3f}".format(o_l[i] / max(1, o_l[i] + o_r[i]))])
        or_.extend([names[i] + ": ", "{:.3f}".format(o_r[i] / max(1, o_l[i] + o_r[i]))])
        os_.extend([names[i] + ": ", "{:.3f}".format(np.sum(o_s[i]) / max(1, wins[i]))])
        lt_.extend([names[i] + ": ", "{:.3f}".format(l_t[i] / (counter[i]))])
        u_s[i] = (u_s[i] / counter[i])
        u_a[i] = (u_a[i] / counter[i])
        us.extend([names[i] + ": ", "{:.3f}".format(np.average(u_s[i]))])
        ua.extend([names[i] + ": ", "{:.3f}".format(np.average(u_a[i]))])
        lat_rmse_.extend([names[i] + ": ", "{:.3f}".format(lat_rmse[i] / (counter[i]))])
        long_rmse_.extend([names[i] + ": ", "{:.3f}".format(long_rmse[i] / (counter[i]))])
        tv_infeasible.extend([names[i] + ": ", "{:.3f}".format(tv_inf[i] / (counter[i]))])
        ego_infeasible.extend([names[i] + ": ", "{:.3f}".format(ego_inf[i] / (counter[i]))])
        d_s_cont[i] = d_s_cont[i] / counter_nc[i]
        col_d_cont[i] = np.log(col_d_cont[i] / counter[i])

        ax1.plot(d_s_cont[i])
        ax2.plot(col_d_cont[i])
        ax3.plot(u_s[i])
        ax4.plot(u_a[i])
    ax1.legend(names)
    ax2.legend(names)
    ax3.legend(names)
    ax4.legend(names)
    ax1.set_ylabel('Average Delta s')
    ax2.set_xlabel('Track progress in %')
    ax2.set_ylabel('Collision metric')
    ax3.set_ylabel('u_s')
    ax4.set_ylabel('u_a')
    print(*names, sep=',')

    print(*avg_delta_s_print)
    print(*w)
    print(*wr)
    print(*wr_nc)
    print(*sr)
    print(*ct)
    print(*ctr)
    print(*ol)
    print(*or_)
    print(*os_)
    print(*lt_)
    print(*ua)
    print(*us)
    print(*ego_infeasible)
    print(*tv_infeasible)
    print(*lat_rmse_, sep=',')
    print(*long_rmse_, sep=',')
    plt.show()
    '''extents = scenario_data.scenario_def.track.track_extents
    xlin = np.array([[np.linspace(extents['x_min'],extents['x_max'], 100)], [np.linspace(extents['x_min'],extents['x_max'], 100)]])'''

    '''fig = plt.figure(figsize=(7, 3))
    ax = plt.gca()
    overtake_pos, xedges, yedges = np.histogram2d([o[0] for o in o_xy[0]], [o[1] for o in o_xy[0]], bins=[45, 45])
    plt.hexbin([o[0] for o in o_xy[0]],[o[1] for o in o_xy[0]], gridsize=30, cmap=plt.cm.jet, bins=None)
    scenario_data.scenario_def.track.plot_map(ax)'''

    plt.subplot(3, 1, 1)
    plt.tight_layout(pad=3)
    plt.hist(o_s[0], bins=20)
    plt.title(names[0])
    plt.ylabel("frequency")
    plt.subplot(3, 1, 2)
    plt.hist(o_s[1], bins=20)
    plt.title(names[1])
    plt.ylabel("frequency")
    plt.subplot(3, 1, 3)
    plt.hist(o_s[2], bins=20)
    plt.title(names[2])
    plt.ylabel("frequency")
    plt.xlabel("Overtake position s")
    plt.show()

    plt.subplot(3, 1, 1)
    plt.tight_layout(pad=3)
    plt.hist(o_s[3], bins=20)
    plt.title(names[3])
    plt.ylabel("frequency")
    plt.subplot(3, 1, 2)
    plt.hist(o_s[4], bins=20)
    plt.title(names[4])
    plt.ylabel("frequency")
    plt.subplot(3, 1, 3)
    plt.hist(o_s[5], bins=20)
    plt.title(names[5])
    plt.ylabel("frequency")
    plt.xlabel("Overtake position s")
    plt.show()
    ## Plotting for lat/long errors
    plt.subplot(1, 2, 1)
    plt.xlabel("Lateral error (m)")
    for i in range(n):
        sns.distplot(lateral_errors[i], color=find_color(names[i]), hist=False)

    plt.legend(names)
    plt.ylim((0, 50))
    plt.xlim((-0.5, 0.5))

    plt.subplot(1, 2, 2)
    plt.xlabel("Longitudinal error (m)")
    for i in range(n):
        sns.distplot(longitudinal_errors[i], color=find_color(names[i]), hist=False)

    plt.ylim((0, 50))
    plt.xlim((-0.5, 0.5))

    plt.suptitle("Comparison of prediction modules \nagainst ground truth trajectory")
    # tikzplotlib.save("rsme")
    plt.show()


if __name__ == '__main__':
    main()