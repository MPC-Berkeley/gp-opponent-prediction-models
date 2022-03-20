#!/usr/bin/env python3

import numpy as np

import time
import warnings
import pdb
from collections import deque
from typing import List, Dict
import os

import multiprocessing as mp
import threading

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

from barcgp.common.tracks.track import get_track
from barcgp.dynamics.models.obstacle_types import BaseObstacle, RectangleObstacle
from barcgp.common.pytypes import VehicleState, Position, VehicleActuation, VehiclePrediction

from barcgp.visualization.vis_types import GlobalPlotConfigs, VehiclePlotConfigs, ObstaclePlotConfigs

import cProfile, pstats, io
from pstats import SortKey

class BarcFigure():
    def __init__(self,
                    t0: float = None,
                    params: GlobalPlotConfigs = GlobalPlotConfigs()) -> None:
        self.track = get_track(params.track_name)
        self.close_loop = params.close_loop

        self.buffer_length = params.buffer_length
        self.draw_period = params.draw_period
        self.plot_title = params.plot_title

        self.t0 = t0

        self.dash_lib = {'solid': None, 'dash': [4,2], 'dot': [2,2], 'dash-dot': [4,2,2,2]}
        self.color_lib = {'r': (255, 0, 0), 'g': (0, 255, 0), 'b': (0, 0, 255)}

        self.vehicle_params = dict()
        self.vehicle_trace_names = dict()
        self.vehicle_trace_styles = dict()
        self.vehicle_rect_vertices = dict()
        self.vehicle_pred = dict()
        self.vehicle_full_traj = dict()
        self.vehicle_ss = dict()
        self.vehicle_ts_data = dict()
        self.vehicle_covariances = dict()
        self.vehicle_pred_covariances = dict()

        self.running = threading.Event() # Signal for keeping I/O thread alive
        self.running.set()
        self.io_thread_init = threading.Event()

        self.plot_queue = dict() # Dict of queues for transferring data between I/O thread and plotter process
        self.new_data = dict() # Events for signaling availability of new data

        self.obstacle_params = dict()
        self.obstacle_shapes = dict()

        self.n_cov_pts = 20
        theta = np.linspace(0, 2*np.pi, self.n_cov_pts)
        self.unit_circle_pts = np.vstack((np.cos(theta), np.sin(theta)))
        self.n_std = 2

        # Mapping from number of upper triangular elements to length of diagonal
        # Used to determin the state dimension given the upper triangular of covariance matrix
        self.triu_len_to_diag_dim = {3: 2, 6: 3, 10: 4, 15: 5, 21: 6}

    # I/O thread
    def send_data_to_queue(self) -> None:
        while self.running.is_set():
            for n in self.vehicle_params.keys():
                if self.new_data[n].is_set(): # Only update if new data is available
                    data = {'rect_vertices': self.vehicle_rect_vertices[n], 'pred': None, 'pred_cov': None, 'ss': None, 'ts': None, 'cov': None}
                    if self.vehicle_params[n].show_pred:
                        data['pred'] = self.vehicle_pred[n]
                        if self.vehicle_params[n].show_cov:
                            data['pred_cov'] = self.vehicle_pred_covariances[n]
                    if self.vehicle_params[n].show_ss:
                        data['ss'] = self.vehicle_ss[n]
                    if self.vehicle_params[n].show_cov:
                        data['cov'] = self.vehicle_covariances[n]

                    self.plot_queue[n].put(data) # Put data into queue for plotting in separate process
                    self.new_data[n].clear() # Clear new data signal

    # This method should be run in a separate process
    def run_plotter(self) -> None:
        app = QtGui.QApplication([])
        pg.setConfigOptions(antialias=True, background='w')

        figsize = (750, 750)
        widget = pg.GraphicsLayoutWidget(show=True)
        widget.setWindowTitle(self.plot_title)
        widget.resize(*figsize)

        # Set up top view x-y plot
        l_xy = widget.addLayout()
        vb_xy = l_xy.addViewBox(lockAspect=True)
        p_xy = l_xy.addPlot(viewBox=vb_xy)
        p_xy.setDownsampling(mode='peak')
        p_xy.setLabel('left', 'Y', units='m')
        p_xy.setLabel('bottom', 'X', units='m')
        track_bbox = self.track.plot_map_qt(p_xy, close_loop=self.close_loop)  # plot the track once at the start and avoid deleting it.
        p_xy.enableAutoRange('xy', False)
        fillLevel = track_bbox[3] + track_bbox[1]

        vehicle_rects = dict()
        vehicle_predicted_trajectories = dict()
        vehicle_predicted_psi = dict()
        vehicle_full_trajectories = dict()
        vehicle_predicted_covariances = dict()
        vehicle_safe_sets = dict()
        vehicle_covariances = dict()
        for n in self.vehicle_params.keys():
            L = self.vehicle_params[n].vehicle_draw_L
            W = self.vehicle_params[n].vehicle_draw_W
            init_V = self.get_rect_verts(0, 0, 0, L, W)
            color_rgb = self.color_lib[self.vehicle_params[n].color]

            # Plot raceline if a file was given
            if self.vehicle_params[n].raceline_file is not None:
                raceline = np.load(os.path.expanduser(self.vehicle_params[n].raceline_file), allow_pickle=True)
                p_xy.plot(raceline['x'], raceline['y'], pen=pg.mkPen(self.vehicle_params[n].color, width=2))

            rects = dict()
            if self.vehicle_params[n].show_sim:
                rects['sim'] = p_xy.plot(init_V[:, 0], init_V[:, 1],
                                         connect='all',
                                         pen=pg.mkPen(self.vehicle_params[n].color, width=1,
                                                      dash=self.dash_lib['solid']),
                                         fillLevel=fillLevel,
                                         fillBrush=color_rgb + (50,))
            if self.vehicle_params[n].show_est:
                rects['est'] = p_xy.plot(init_V[:, 0], init_V[:, 1],
                                         connect='all',
                                         pen=pg.mkPen(self.vehicle_params[n].color, width=1,
                                                      dash=self.dash_lib['dash']),
                                         fillLevel=fillLevel,
                                         fillBrush=color_rgb + (50,))
            if self.vehicle_params[n].show_mea:
                rects['mea'] = p_xy.plot(init_V[:, 0], init_V[:, 1],
                                         connect='all',
                                         pen=pg.mkPen(self.vehicle_params[n].color, width=1, dash=self.dash_lib['dot']),
                                         fillLevel=fillLevel,
                                         fillBrush=color_rgb + (50,))

            vehicle_rects[n] = rects

            # Add an empty line plot for each predicted trajectory
            if self.vehicle_params[n].show_full_traj:
                vehicle_full_trajectories[n] = p_xy.plot([], [], pen=self.vehicle_params[n].color,
                                                         symbolBrush=self.vehicle_params[n].color, symbolPen='k',
                                                         symbol='o', symbolSize=2)
                self.prepare_full_traj(n, self.vehicle_params[n].state_list)
            if self.vehicle_params[n].show_pred:
                vehicle_predicted_trajectories[n] = p_xy.plot([], [], pen=self.vehicle_params[n].color,
                                                              symbolBrush=self.vehicle_params[n].color, symbolPen='k',
                                                              symbol='o', symbolSize=5)
                vehicle_predicted_psi[n] = p_xy.plot([], [],
                                                     connect='all',
                                                     pen=pg.mkPen(self.vehicle_params[n].color, width=1,
                                                                  dash=self.dash_lib['solid']),
                                                     symbolBrush=None, symbolPen=None)
                vehicle_predicted_covariances[n] = [p_xy.plot([], [],
                                                              pen=pg.mkPen(self.vehicle_params[n].color, width=1e-3,
                                                                           dash=self.dash_lib['solid']),
                                                              fillLevel=fillLevel, fillBrush=color_rgb + (50,)) for _ in
                                                    range(self.n_cov_pts)]
            if self.vehicle_params[n].show_ss:
                vehicle_safe_sets[n] = p_xy.plot([], [], pen=None, symbolBrush=None,
                                                 symbolPen=self.vehicle_params[n].color, symbol='s', symbolSize=6)

            # Add an empty line plot for each covariance ellipse
            if self.vehicle_params[n].show_cov:
                vehicle_covariances[n] = p_xy.plot([], [], pen=self.vehicle_params[n].color, symbolBrush=None,
                                                   symbolPen=None)

        # redraw method is attached to timer with period self.draw_period
        def redraw():
            for n in self.vehicle_params.keys():
                if not self.plot_queue[n].empty(): # Don't redraw if data queue is empty
                    while not self.plot_queue[n].empty(): # Read queue until empty
                        data = self.plot_queue[n].get()
                    rect_vertices = data['rect_vertices']
                    if 'sim' in rect_vertices.keys():
                        if rect_vertices['sim'] is not None:
                            vehicle_rects[n]['sim'].setData(rect_vertices['sim'][:,0], rect_vertices['sim'][:,1])
                    if 'est' in rect_vertices.keys():
                        if rect_vertices['est'] is not None:
                            vehicle_rects[n]['est'].setData(rect_vertices['est'][:,0], rect_vertices['est'][:,1])
                    if 'mea' in rect_vertices.keys():
                        if rect_vertices['mea'] is not None:
                            vehicle_rects[n]['mea'].setData(rect_vertices['mea'][:,0], rect_vertices['mea'][:,1])
                    # p_xy.setRange(rect=view_rect)

                    ts_data = data['ts']
                    if ts_data is not None:
                        if ts_data['state']:
                            for s in subplots['state']:
                                s.update(n, ts_data['state'])
                                s.redraw()
                        if ts_data['input']:
                            for s in subplots['input']:
                                s.update(n, ts_data['input'])
                                s.redraw()

                    pred = data['pred']
                    if pred is not None:
                        vehicle_predicted_trajectories[n].setData(pred['x'], pred['y'])
                        if self.vehicle_params[n].show_full_vehicle_bodies:
                            x_s = []; y_s = []
                            for i in range(len(pred['x'])):
                                a = self.get_rect_verts(pred['x'][i], pred['y'][i], pred['psi'][i], L, W)
                                x_s.extend(a[:, 0])
                                y_s.extend(a[:, 1])
                            vehicle_predicted_psi[n].setData(x_s, y_s)

                        pred_cov = data['pred_cov']
                        if pred_cov is not None:
                            for (i, pc) in enumerate(pred_cov):
                                vehicle_predicted_covariances[n][i].setData(pc['x'], pc['y'])
                    if self.vehicle_params[n].show_full_traj:
                        vehicle_full_trajectories[n].setData(self.vehicle_full_traj[n]['x'],self.vehicle_full_traj[n]['y'])
                    ss = data['ss']
                    if ss is not None:
                        vehicle_safe_sets[n].setData(ss['x'], ss['y'])

                    cov = data['cov']
                    if cov is not None:
                        vehicle_covariances[n].setData(cov['x'], cov['y'])

        timer = QtCore.QTimer()
        timer.timeout.connect(redraw)
        timer.start(self.draw_period*1000)

        app.exec_()

        return

    def add_vehicle(self,
                params: VehiclePlotConfigs = VehiclePlotConfigs()) -> str:

        L = params.vehicle_draw_L
        W = params.vehicle_draw_W
        name = params.name

        self.vehicle_params[name] = params

        init_V = self.get_rect_verts(0, 0, 0, L, W)

        trace_names = {'state': [], 'input': []}
        trace_styles= {'state': [], 'input': []}
        rect_vertices = dict()
        if params.show_sim:
            rect_vertices['sim'] = init_V
            trace_names['state'].append('sim')
            trace_styles['state'].append('solid')
        if params.show_est:
            rect_vertices['est'] = init_V
            trace_names['state'].append('est')
            trace_styles['state'].append('dash')
        if params.show_mea:
            rect_vertices['mea'] = init_V
            trace_names['state'].append('mea')
            trace_styles['state'].append('dot')
        if params.show_ecu:
            trace_names['input'].append('ecu')
            trace_styles['input'].append('solid')

        self.vehicle_trace_names[name] = trace_names
        self.vehicle_trace_styles[name] = trace_styles
        self.vehicle_rect_vertices[name] = rect_vertices

        if params.show_cov:
            self.vehicle_covariances[name] = {'x': [], 'y': []}

        if params.show_full_traj:
            self.vehicle_full_traj[name] = {'x': [], 'y': [], 'psi': []}

        if params.show_pred:
            self.vehicle_pred[name] = {'x': [], 'y': [], 'psi': []}
            if params.show_cov:
                self.vehicle_pred_covariances[name] = [{'x': [], 'y': []}]

        if params.show_ss:
            self.vehicle_ss[name] = {'x': [], 'y': [], 'psi': []}

        self.plot_queue[name] = mp.Queue() # Create queue for vehicle plot data
        self.new_data[name] = threading.Event() # Create event to signal new data for vehicle

        return name

    def update_vehicle_state(self, vehicle_name: str,
                  sim_data: VehicleState,
                  mea_data: VehicleState,
                  est_data: VehicleState,
                  ecu_data: VehicleActuation) -> None:
        if vehicle_name in self.vehicle_params.keys():
            ts_data = {'state': dict(), 'input': dict()}
            rect_vertices = dict()
            xy_cov_ell_pts = np.empty((0, 2))

            L = self.vehicle_params[vehicle_name].vehicle_draw_L
            W = self.vehicle_params[vehicle_name].vehicle_draw_W

            if self.vehicle_params[vehicle_name].show_sim:
                rect_vertices['sim'] = None
                ts_data['state']['sim'] = None
                if sim_data.t is not None:
                    if None in (sim_data.x.x, sim_data.x.y, sim_data.e.psi):
                        sim_data.x.x, sim_data.x.y, sim_data.e.psi = self.track.local_to_global([sim_data.p.s, sim_data.p.x_tran, sim_data.p.e_psi])
                    rect_vertices['sim'] = self.get_rect_verts(sim_data.x.x, sim_data.x.y, sim_data.e.psi, L, W)
                    ts_data['state']['sim'] = sim_data
            if self.vehicle_params[vehicle_name].show_mea:
                rect_vertices['mea'] = None
                ts_data['state']['mea'] = None
                if mea_data.t is not None:
                    if None in (mea_data.x.x, mea_data.x.y, mea_data.e.psi):
                        mea_data.x.x, mea_data.x.y, mea_data.e.psi = self.track.local_to_global([mea_data.p.s, mea_data.p.x_tran, mea_data.p.e_psi])
                    rect_vertices['mea'] = self.get_rect_verts(mea_data.x.x, mea_data.x.y, mea_data.e.psi, L, W)
                    ts_data['state']['mea'] = mea_data
            if self.vehicle_params[vehicle_name].show_est:
                rect_vertices['est'] = None
                ts_data['state']['est'] = None
                if est_data.t is not None:
                    if None in (est_data.x.x, est_data.x.y, est_data.e.psi):
                        est_data.x.x, est_data.x.y, est_data.e.psi = self.track.local_to_global([est_data.p.s, est_data.p.x_tran, est_data.p.e_psi])
                    rect_vertices['est'] = self.get_rect_verts(est_data.x.x, est_data.x.y, est_data.e.psi, L, W)
                    ts_data['state']['est'] = est_data

            self.vehicle_rect_vertices[vehicle_name] = rect_vertices

            if self.vehicle_params[vehicle_name].show_ecu:
                ts_data['input']['ecu'] = None
                if ecu_data.t is not None:
                    ts_data['input']['ecu'] = ecu_data

            self.vehicle_ts_data[vehicle_name] = ts_data
        else:
            warnings.warn("Vehicle name '%s' not recognized, nothing to update..." % vehicle_name, UserWarning)

    def prepare_full_traj(self, vehicle_name: str, listOfStates: List[VehicleState]):
        x_st, y_st, psi_st = [], [], []
        for i in range(len(listOfStates)):
            (x, y, psi) = listOfStates[i].x.x, listOfStates[i].x.y, listOfStates[i].e.psi
            x_st.append(x)
            y_st.append(y)
            psi_st.append(psi)
        self.vehicle_full_traj[vehicle_name]['x'], self.vehicle_full_traj[vehicle_name]['y'], self.vehicle_full_traj[vehicle_name]['psi'] = x_st, y_st, psi_st

    def update_vehicle_prediction(self, vehicle_name: str, pred_data: VehiclePrediction) -> None:
        if pred_data.x is None or pred_data.y is None or len(pred_data.x) == 0 or len(pred_data.y) == 0: # x, y, or psi fields are empty
            x_pred, y_pred, psi_pred = [], [], []
            for i in range(len(pred_data.s)):
                (x, y, psi) = self.track.local_to_global([pred_data.s[i], pred_data.x_tran[i], pred_data.e_psi[i]])
                x_pred.append(x); y_pred.append(y); psi_pred.append(psi)
        else:
            x_pred, y_pred, psi_pred = pred_data.x, pred_data.y, pred_data.psi
        self.vehicle_pred[vehicle_name]['x'], self.vehicle_pred[vehicle_name]['y'], self.vehicle_pred[vehicle_name]['psi'] = x_pred, y_pred, psi_pred

        # Update covariance ellipses over prediction horizon
        if self.vehicle_params[vehicle_name].show_cov:
            N = len(x_pred)
            ell_pts = [None for _ in range(N-1)]
            if pred_data.xy_cov is not None:
                for i in range(1, N):
                    angle = psi_pred[i] # + pred_data.e_psi[i]
                    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
                    cov = pred_data.xy_cov[i]
                    # print("bodycov", cov)
                    cov_ell_pts = (self.n_std * np.sqrt(
                        cov).dot(self.unit_circle_pts).T).transpose()

                    cov_ell_pts = (rot_mat@cov_ell_pts).transpose()
                    xy_cov_ell_pts = np.array([x_pred[i], y_pred[i]]) + cov_ell_pts

                    ell_pts[i - 1] = {'x': xy_cov_ell_pts[:, 0], 'y': xy_cov_ell_pts[:, 1]}
            elif pred_data.sey_cov:
                sey_unflat = np.array(pred_data.sey_cov).reshape(N, 4)
                for i in range(1, N):
                    sey_cov = sey_unflat[i].reshape(2, 2)
                    print(sey_cov)
                    sey_cov_ell_pts = np.array([pred_data.s[i], pred_data.x_tran[i]]) + self.n_std * np.linalg.cholesky(
                        sey_cov).dot(self.unit_circle_pts).T
                    xy_cov_ell_pts = np.array([self.track.local_to_global([sey_cov_ell_pts[j][0],sey_cov_ell_pts[j][1], 0])[:2] for j in
                                               range(self.n_cov_pts)])
                    ell_pts[i - 1] = {'x': xy_cov_ell_pts[:, 0], 'y': xy_cov_ell_pts[:, 1]}

            self.vehicle_pred_covariances[vehicle_name] = ell_pts

    def update_vehicle_safe_set(self, vehicle_name: str, ss_data: VehiclePrediction) -> None:
        if not ss_data.x or not ss_data.y or not ss_data.psi:
            x_ss, y_ss, psi_ss = [], [], []
            for i in range(len(ss_data.s)):
                (x, y, psi) = self.track.local_to_global([ss_data.s[i], ss_data.x_tran[i], ss_data.e_psi[i]])
                x_ss.append(x); y_ss.append(y); psi_ss.append(psi)
        else:
            x_ss, y_ss, psi_ss = ss_data.x, ss_data.y, ss_data.psi
        self.vehicle_ss[vehicle_name]['x'], self.vehicle_ss[vehicle_name]['y'], self.vehicle_ss[vehicle_name]['psi'] = x_ss, y_ss, psi_ss

    def update(self, vehicle_name: str,
                sim_data: VehicleState = VehicleState(),
                mea_data: VehicleState = VehicleState(),
                est_data: VehicleState = VehicleState(),
                ecu_data: VehicleActuation = VehicleActuation(),
                pred_data: VehiclePrediction = VehiclePrediction(),
                ss_data: VehiclePrediction = VehiclePrediction()) -> None:

        if vehicle_name in self.vehicle_params.keys():
            self.update_vehicle_state(vehicle_name, sim_data, mea_data, est_data, ecu_data)
            if self.vehicle_params[vehicle_name].show_pred and pred_data.t is not None:
                self.update_vehicle_prediction(vehicle_name, pred_data)
            if self.vehicle_params[vehicle_name].show_ss and ss_data.t is not None:
                self.update_vehicle_safe_set(vehicle_name, ss_data)
            self.new_data[vehicle_name].set() # Signal update thread that new data is ready
        else:
            warnings.warn("Vehicle name '%s' not recognized, nothing to update..." % vehicle_name, UserWarning)

        # Start I/O thread on first call of update
        if not self.io_thread_init.is_set():
            self.update_thread = threading.Thread(target=self.send_data_to_queue)
            self.update_thread.start()
            self.io_thread_init.set()

    def get_rect_verts(self, x_center: float, y_center: float, theta: float, L: float, W: float) -> np.ndarray:
        if None in (x_center, y_center, theta, L, W):
            return None

        tr = [x_center + L*np.cos(theta)/2 - W*np.sin(theta)/2, y_center + L*np.sin(theta)/2 + W*np.cos(theta)/2]
        br = [x_center + L*np.cos(theta)/2 + W*np.sin(theta)/2, y_center + L*np.sin(theta)/2 - W*np.cos(theta)/2]
        tl = [x_center - L*np.cos(theta)/2 - W*np.sin(theta)/2, y_center - L*np.sin(theta)/2 + W*np.cos(theta)/2]
        bl = [x_center - L*np.cos(theta)/2 + W*np.sin(theta)/2, y_center - L*np.sin(theta)/2 - W*np.cos(theta)/2]

        return np.array([bl, br, tr, tl, bl])

    # def add_obstacle(self, rect: BaseObstacle,
    #             params: ObstaclePlotConfigs = ObstaclePlotConfigs()) -> None:
    #     name = params.name
    #     self.obstacle_params[name] = params
    #     self.obstacle_shapes[name] = patches.Polygon(rect.xy, facecolor = params.color, alpha = params.alpha)
    #     self.ax.add_patch(self.obstacle_shapes[name])
    #     return

    # def update_obstacle(self, obstacle_name: str, obs: BaseObstacle) -> None:
    #     if obstacle_name in self.obstacle_params.keys():
    #         self.obstacle_shapes[obstacle_name].set_xy(obs.xy)
    #     return

    # def remove_vehicle(vehicle_name: str) -> None:
    #     #TODO: Not yet implemented
    #     return

    # def remove_obstacle(self, obstacle_name: str) -> None:
    #     if obstacle_name in self.obstacle_params.keys():
    #         self.obstacle_params.pop(obstacle_name)
    #         self.obstacle_shapes[obstacle_name].remove()
    #         self.obstacle_shapes.pop(obstacle_name)
    #     return

class timeHistorySubplot():
    '''
    creates a scrollingPlot() objects and adds data to it from structures
    adds the variabe 'var' from data objects passed to it

    ex: if var = 'x', it will plot sim_data.x, est_data.x, and mea_data.x
    '''
    def __init__(self, p: pg.PlotDataItem, var: str, units: str = None, t0: float = None):
        self.p = p
        self.plotters = dict()
        self.plotter_trace_names = dict()
        self.var = var
        self.t0 = t0

        self.p.setLabel('left', var, units=units)

    '''
    Adds a scrollingPlot to the subplot. The plot may contain multiple traces, whose
    names are specified in the list trace_names
    '''
    def add_plot(self, n_pts: int,
                plot_name: str = 'plot_1',
                trace_names: List[str] = ['trace_1'],
                trace_styles: List[str] = ['-'],
                trace_color: str = 'b') -> None:
        if len(trace_styles) < len(trace_names):
            trace_styles.extend(['-' for _ in range(len(trace_names)-len(trace_styles))])
            warnings.warn("Should specify line styles for each trace, using default '-' for unspecified traces", UserWarning)
        if plot_name not in self.plotters.keys():
            plotter = scrollingPlot(self.p, n_pts, trace_names, trace_styles, trace_color, t0=self.t0)
            self.plotters[plot_name] = plotter
            self.plotter_trace_names[plot_name] = trace_names
        else:
            warnings.warn("Plot with the name '%s' already exists, not adding..." % plot_name, UserWarning)

    def update(self, plot_name: str, data: dict) -> None:
        # adds data to the plot, which will remove oldest data points
        # pass 'None' type to avoid updating
        if plot_name in self.plotters.keys() and data:
            for n in data.keys():
                if n in self.plotter_trace_names[plot_name]:
                    if data[n] is not None:
                        if '.' in self.var:
                            field_names = self.var.split('.')
                            d = data[n]
                            for s in field_names:
                                d = getattr(d, s)
                        else:
                            d = getattr(data[n], self.var)
                        self.plotters[plot_name].add_point(data[n].t, d, trace_name=n)
                else:
                    warnings.warn("Trace name '%s' not recognized, nothing to update..." % n, UserWarning)
        else:
           warnings.warn("Plot name '%s' not recognized, nothing to update..." % plot_name, UserWarning)

    def redraw(self) -> None:
        for p in self.plotters.values():
            p.redraw()

class scrollingPlot():
    '''
    Creates and updates a scrolling plot on 'ax'
    num_pts: size of the list to be used for each trace
    num_traces: number of different input to plot

    Example: when comparing simulated and estimated velocity, there would be two traces
    and however many prior points are desired (perhaps 50)
    '''
    def __init__(self, p: pg.PlotDataItem, num_pts: int,
                trace_names: List[str] = ['trace_1'],
                trace_styles: List[str] = ['solid'],
                trace_color: str = 'b',
                t0: float = None):
        '''
        Warning: time.time() in a function argument corresponds to the time when the function was created
        (when the python interpreter started)
        It is only suitable for initializers!!!!!!! Do not use for methods used to update.
        '''

        self.p = p
        # self.pts = num_pts
        self.trace_names = trace_names
        self.t0 = t0
        self.init = True

        dash_lib = {'solid': None, 'dash': [4,2], 'dot': [2,2], 'dash-dot': [4,2,2,2]}
        #WARNING: anything of this form will wind up with pass-by-reference issues
        #self.pt_memory   = [[None]*num_pts]*num_traces
        #self.time_memory = [[None]*num_pts]*num_traces

        self.pt_memory = dict()
        self.time_memory = dict()
        self.lines = dict()
        for (n, s) in zip(trace_names, trace_styles):
            self.pt_memory[n] = deque([], num_pts)
            self.time_memory[n] = deque([], num_pts)
            self.lines[n] = self.p.plot(self.time_memory[n], self.pt_memory[n],
                                        pen=pg.mkPen(trace_color, width=1, dash=dash_lib[s]))

    def add_point(self, t: float, x: float, trace_name: str) -> None:
        if trace_name not in self.trace_names:
            return
        if x is None or t is None or t < 0:
            return

        if self.t0 is not None:
            t = t - self.t0

        self.pt_memory[trace_name].append(x)
        self.time_memory[trace_name].append(t)

        return

    def redraw(self) -> None:
        for n in self.trace_names:
            self.lines[n].setData(self.time_memory[n], self.pt_memory[n])
