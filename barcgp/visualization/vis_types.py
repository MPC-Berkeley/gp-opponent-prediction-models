from dataclasses import dataclass, field
import time

from barcgp.common.pytypes import VehicleState
from typing import List

import numpy as np

from barcgp.common.pytypes import PythonMsg

@dataclass
class GlobalPlotConfigs(PythonMsg):
    # Global visualization params
    track_name: str         = field(default = 'Monza_Track')

    show_warnings: bool     = field(default = True)

    draw_period: float      = field(default = 0.1)
    update_period: float    = field(default = 0.001)

    buffer_length: int      = field(default = 100)

    plot_title: str         = field(default = 'BARC Plotter')

    close_loop: bool        = field(default = True)

@dataclass
class VehiclePlotConfigs(PythonMsg):
    # Vehicle specific params
    name: str               = field(default = 'barc_1')
    color: str              = field(default = 'b')

    show_sim: bool          = field(default = False)
    show_mea: bool          = field(default = False)
    show_est: bool          = field(default = True)
    show_ecu: bool          = field(default = True)
    show_pred: bool         = field(default = False)
    show_traces: bool       = field(default = True)
    show_full_traj: bool    = field(default = False)
    state_list: List[VehicleState] = field(default = List[VehicleState])
    show_ss: bool           = field(default = False)
    show_cov: bool          = field(default = False)
    show_full_vehicle_bodies: bool          = field(default = False)

    vehicle_draw_L: float   = field(default = 0.25)
    vehicle_draw_W: float   = field(default = 0.1)

    simulated: bool         = field(default = False)

    raceline_file: str      = field(default = None)

@dataclass
class ObstaclePlotConfigs(PythonMsg):
    name: str               = field(default = 'obs_1')
    color: str              = field(default = 'b')

    alpha: float            = field(default = 1)
