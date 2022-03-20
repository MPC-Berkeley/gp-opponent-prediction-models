import numpy as np

from barcgp.dynamics.models.dynamics_models import get_dynamics_model
from barcgp.common.pytypes import VehicleState

class DynamicsSimulator():
    '''
    Class for simulating vehicle dynamics, default settings are those for BARC

    '''
    def __init__(self, t0: float, dynamics_config, track=None):
        self.model = get_dynamics_model(t0, dynamics_config, track=track)
        return

    def step(self, state: VehicleState):
        #update the vehicle state
        self.model.step(state)
