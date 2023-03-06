#!/usr/bin/env python3
"""
Script that evaluates and visualizes post-processed data
"""
from barcgp.common.utils.file_utils import *
import os
import colorsys
import numpy as np
import seaborn as sns
from barcgp.common.utils.scenario_utils import PostprocessData
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
import matplotlib.colors as mcolors

post_data = os.path.join(eval_dir, 'aggressive_blocking/aggressive_blocking.pkl')

def print_overview(data):
    avg_delta_s_print = ["AVG Delta S - "]
    w = ["Wins - "]
    wr = ["Win Rate - "]
    wr_nc = ["Win Rate nc - "]
    ct = ["Crashes total - "]
    ctr = ["Crashes total rate - "]
    lt_ = ["Left track rate - "]
    us = ["AVG u_s - "]
    ua = ["AVG u_a - "]
    ua_min = ["AVG Min u_a - "]
    '''lat_rmse_ = ["Lateral RMSE - "]
    long_rmse_ = ["Longitudinal RMSE - "]'''
    ego_infeasible = ["Ego Infeas - "]
    tv_infeasible = ["TV Infeas - "]
    for i in data:
        avg_delta_s_print.extend([i + ": ", "{:.3f}".format(data[i].avg_delta_s)])
        w.extend([i + ": ", data[i].num_wins])
        wr.extend([i + ": ", "{:.3f}".format(data[i].win_rate)])
        wr_nc.extend([i + ": ", "{:.3f}".format(data[i].win_rate_nc)])
        ct.extend([i + ": ", data[i].num_crashes])
        ctr.extend([i + ": ", "{:.3f}".format(data[i].crash_rate)])
        lt_.extend([i + ": ", "{:.3f}".format(data[i].left_track_rate)])
        us.extend([i + ": ", "{:.3f}".format(data[i].avg_abs_steer)])
        ua.extend([i + ": ", "{:.3f}".format(data[i].avg_a)])
        ua_min.extend([i + ": ", "{:.3f}".format(data[i].avg_min_a)])
        tv_infeasible.extend([i + ": ", "{:.3f}".format(data[i].tv_inf_rate)])
        ego_infeasible.extend([i + ": ", "{:.3f}".format(data[i].ego_inf_rate)])
    print(*avg_delta_s_print)
    print(*w)
    print(*wr)
    print(*wr_nc)
    print(*ct)
    print(*ctr)
    print(*lt_)
    print(*us)
    print(*ua)
    print(*ua_min)
    print(*ego_infeasible)
    print(*tv_infeasible)


def plot_heatmap(data_win, data_crash, color_map_win, color_map_crash, track):
    fig, ax = plt.subplots()
    track.plot_map(ax)
    heatmap, xedges, yedges = np.histogram2d(data_win[0], data_win[1], range=(
        [track.track_extents['x_min'], track.track_extents['x_max']],
        [track.track_extents['y_min'], track.track_extents['y_max']]), bins=1000)
    heatmap_c, xedges_c, yedges_c = np.histogram2d(data_crash[0], data_crash[1], range=(
        [track.track_extents['x_min'], track.track_extents['x_max']],
        [track.track_extents['y_min'], track.track_extents['y_max']]), bins=1000)
    heatmap = gaussian_filter(heatmap, sigma=10)
    heatmap_c = gaussian_filter(heatmap_c, sigma=10)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=color_map_win)
    ax.imshow(heatmap_c.T, extent=extent, origin='lower', cmap=color_map_crash)
    ax.legend()
    plt.show()


def get_colors(data):
    colors = []
    colors.append([0, 1, 1])
    colors.append([243 / 360, 1, 1])
    colors.append([123 / 360, 1, 1])
    num_classes = 0
    found_prefixes = []
    color_sets = []
    for i in data:
        if not i[0] in found_prefixes:
            num_classes += 1
            found_prefixes.append(i[0])
    n = int(len(data) / num_classes)
    for i in range(num_classes):
        for k in range(n):
            color_sets.append(colorsys.hsv_to_rgb(colors[i][0], colors[i][1] - k / n, colors[i][2]))
    return color_sets, num_classes


def plot_pred_accuracy(data, color_sets, num_classes):
    names = [i for i in data]
    plt.subplot(1, 2, 1)
    plt.xlabel("Lateral error (m)")
    # TODO Merge Classes
    for id, i in enumerate(names):
        # plt.hist(np.array(data[i].lateral_errors), color=[color_sets[id][0], color_sets[id][1], color_sets[id][2], 0.5], bins=100)
        sns.distplot(data[i].lateral_errors, color=color_sets[id], hist=False)
    plt.legend(names)
    '''plt.ylim((0, 50))
    plt.xlim((-0.5, 0.5))'''

    plt.subplot(1, 2, 2)
    plt.xlabel("Longitudinal error (m)")
    for id, i in enumerate(names):
        # plt.hist(np.array(data[i].longitudinal_errors), color=[color_sets[id][0], color_sets[id][1], color_sets[id][2], 0.5], bins=100)
        sns.distplot(data[i].longitudinal_errors, color=color_sets[id], hist=False)
    plt.show()

def compare(run1:PostprocessData, run2:PostprocessData, wins=True):
    if wins:
        return list(set(run1.win_ids).difference(run2.win_ids))
    return list(set(run1.crash_ids).difference(run2.crash_ids))

def main(args=None):
    data = pickle_read(post_data)
    color_sets, num_classes = get_colors(data)
    print_overview(data)
    track = next(iter(data.values())).track
    colors = [(1, 0, 0, 2*(-0.5 + 1/(1 + np.exp(-c)))) for c in np.linspace(0, 20, 100)]
    cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)
    colors = [(0, 0, 1, c) for c in np.linspace(0, 1, 100)]

    cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)
    test_name = 'NLMPC_01'
    plot_heatmap([data[test_name].overtake_x, data[test_name].overtake_y], [data[test_name].crash_x, data[test_name].crash_y],
                 cmapblue, cmapred, track)
    plot_pred_accuracy(data, color_sets, num_classes)
    a = compare(data['NLMPC_01'], data['GP_5 '], wins=True)
    print("NLMPC Win but not GP win:", len(a), a)
    print(data['GP_5 '].crash_ids_ego)
    print(len(data['GP_5 '].crash_ids_ego), len(data['GP_5 '].crash_ids_tv))
    print(len(data['NLMPC_01'].crash_ids_ego), len(data['NLMPC_01'].crash_ids_tv))
    # print(compare(data['GP01'], data['NLMPC_01']))
    b = set(a).difference(data['GP_5 '].crash_ids_tv)
    print("NLMPC Win but not GP win without GP crashes:", len(b))
    print(b)
    print("GP Win but not NLMPC win without NLMPC crashes:", len(set(compare(data['GP_5 '], data['NLMPC_01'], wins=True)).difference(data['NLMPC_01'].crash_ids_tv)))


if __name__ == '__main__':
    main()