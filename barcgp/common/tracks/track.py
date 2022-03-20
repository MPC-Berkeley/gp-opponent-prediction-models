import numpy as np
from numpy import linalg as la

from barcgp.common.tracks.radius_arclength_track import RadiusArclengthTrack

from matplotlib import pyplot as plt
import pdb
import os


def get_save_folder():
    return os.path.join(os.path.dirname(__file__), 'track_data')


def get_available_tracks():
    save_folder = get_save_folder()
    return os.listdir(save_folder)


def get_track(track_file):
    if not track_file.endswith('.npz'):
        track_file += '.npz'

    if track_file not in get_available_tracks():
        raise ValueError('Chosen Track is unavailable: %s\nlooking in:%s\n Available Tracks: %s' % (track_file,
                                                                                                    os.path.join(os.path.dirname(
                                                                                                        __file__), 'tracks', 'track_data'),
                                                                                                    str(get_available_tracks())))

    save_folder = get_save_folder()
    load_file = os.path.join(save_folder, track_file)

    npzfile = np.load(load_file, allow_pickle=True)
    if npzfile['save_mode'] == 'radius_and_arc_length':
        track = RadiusArclengthTrack()
        track.initialize(npzfile['track_width'],
                         npzfile['slack'], npzfile['cl_segs'])
    else:
        raise NotImplementedError(
            'Unknown track save mode: %s' % npzfile['save_mode'])

    return track


def generate_curvature_and_path_length_track(filename, track_width, cl_segs, slack):
    save_folder = get_save_folder()
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, filename)
    np.savez(save_path, save_mode='radius_and_arc_length',
             track_width=track_width, cl_segs=cl_segs, slack=slack)
    return


def generate_Monza_track():
    track_width = 1.5
    slack = 0.45  # Don't change?

    straight_01 = np.array([[3.0, 0]])

    enter_straight_length = 0.5
    curve_length = 3
    curve_swept_angle = np.pi - np.pi/40
    s = 1
    exit_straight_length = 2.2
    curve_11 = np.array([
                        [curve_length, s * curve_length / curve_swept_angle],
                        [exit_straight_length, 0]])

    # WACK CURVE

    enter_straight_length = 0.2
    curve_length = 0.8
    curve_swept_angle = np.pi / 4
    s = -1
    exit_straight_length = 0.4
    curve_10 = np.array([
        [curve_length, s * curve_length / curve_swept_angle],
        [exit_straight_length, 0]])

    enter_straight_length = 0.05
    curve_length = 1.2
    curve_swept_angle = np.pi / 4
    s = 1
    exit_straight_length = 0.4
    curve_09 = np.array([
        [curve_length, s * curve_length / curve_swept_angle],
        [exit_straight_length, 0]])

    enter_straight_length = 0.05
    curve_length = 0.8
    curve_swept_angle = np.pi / 4
    s = -1
    exit_straight_length = 2.5
    curve_08 = np.array([
        [curve_length, s * curve_length / curve_swept_angle],
        [exit_straight_length, 0]])

    # Curve mini before 07
    enter_straight_length = 0
    curve_length = 1.0
    curve_swept_angle = np.pi / 12
    s = -1
    exit_straight_length = 1.5
    curve_before_07 = np.array([
        [curve_length, s * curve_length / curve_swept_angle],
        [exit_straight_length, 0]])

    # Curve 07
    enter_straight_length = 0
    curve_length = 1.2
    curve_swept_angle = np.pi / 3
    s = 1
    exit_straight_length = 1.5
    curve_07 = np.array([[curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    # Curve 06
    enter_straight_length = 0
    curve_length = 1.5
    curve_swept_angle = np.pi / 2
    s = 1
    exit_straight_length = 2.0
    curve_06 = np.array([
        [curve_length, s * curve_length / curve_swept_angle],
        [exit_straight_length, 0]])

    # Chicane 05
    enter_straight_length = 0
    curve1_length = 0.4
    s1, s2 = 1, -1
    curve1_swept_angle = np.pi/8
    mid_straight_length = 1.0
    curve2_length = 0.4
    curve2_swept_angle = np.pi/8
    exit_straight_length = 1.0 + \
        np.abs(np.sin(-1.6493361431346418) *
               (0.4299848245548139+0.0026469133545887783))

    chicane_05 = np.array([
        [curve1_length, s1 * curve1_length / curve1_swept_angle],
        [mid_straight_length, 0],
        [curve2_length, s2 * curve2_length / curve2_swept_angle],
        [exit_straight_length, 0]])

    # Curve 03
    enter_straight_length = 0.0
    curve_length = 4.0
    curve_swept_angle = np.pi / 2 + np.pi/16
    s = 1
    exit_straight_length = 2.0
    curve_03 = np.array([
        [curve_length, s * curve_length / curve_swept_angle],
        [exit_straight_length, 0]])

    # Curve 02
    enter_straight_length = 0.0
    curve_length = 1.0
    curve_swept_angle = np.pi / 10
    s = -1
    exit_straight_length = 1.0
    curve_02 = np.array([
        [curve_length, s * curve_length / curve_swept_angle],
        [exit_straight_length, 0]])

    # Final curve
    curve_length = 1.0
    curve_swept_angle = -np.pi / 10 + 0.11780972450961658
    exit_straight_length = 1.0 - 1.26433096e-01 + 0.0002070341780330276 + \
        0.00021382215942933325 + 1.6293947847880575e-05 - 0.00023011610727452503
    s = -1
    curve_Final = np.array([
        [curve_length, s * curve_length / curve_swept_angle],
        [exit_straight_length, 0]])

    cl_segs = []
    cl_segs.extend(straight_01)
    cl_segs.extend(curve_11)
    cl_segs.extend(curve_10)
    cl_segs.extend(curve_09)
    cl_segs.extend(curve_08)
    cl_segs.extend(curve_before_07)
    cl_segs.extend(curve_07)
    cl_segs.extend(curve_06)
    cl_segs.extend(chicane_05)
    cl_segs.extend(curve_03)
    cl_segs.extend(curve_02)
    cl_segs.extend(curve_Final)

    generate_curvature_and_path_length_track(
        'Monza_Track', track_width, cl_segs, slack)
    print('Generated Monza_Track')
    return
