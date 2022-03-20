#!/usr/bin python3

from dataclasses import dataclass, field
import numpy as np

from barcgp.common.pytypes import PythonMsg, VehicleState

@dataclass
class ControllerConfig(PythonMsg):
    dt: float = field(default=0.1)

@dataclass
class PIDParams(ControllerConfig):
    Kp: float = field(default=2.0)
    Ki: float = field(default=0.0)
    Kd: float = field(default=0.0)

    int_e_max: float = field(default=100)
    int_e_min: float = field(default=-100)
    u_max: float = field(default=None)
    u_min: float = field(default=None)
    du_max: float = field(default=None)
    du_min: float = field(default=None)

    noise: bool = field(default=False)
    noise_max: float = field(default=None)
    noise_min: float = field(default=None)

    periodic_disturbance: bool = field(default=False)
    disturbance_amplitude: float = field(default=None)
    disturbance_period: float = field(default=None)

    def default_speed_params(self):
        self.Kp = 1
        self.Ki = 0
        self.Kd = 0
        self.u_min = -2
        self.u_max = 2
        self.du_min = -10 * self.dt
        self.du_max =  10 * self.dt
        self.noise = False
        return

    def default_steer_params(self):
        self.Kp = 1
        self.Ki = 0.0005 / self.dt
        self.Kd = 0
        self.u_min = -0.35
        self.u_max = 0.35
        self.du_min = -4 * self.dt
        self.du_max = 4 * self.dt
        self.noise = False
        return

@dataclass
class NLMPCParams(ControllerConfig):
    optlevel: int = field(default=1)
    slack: bool = field(default=True)
    solver_dir: str = field(default='')

    n: int = field(default=6) # dimension state space
    d: int = field(default=2) # dimension input space

    N: int = field(default=10) # horizon length

    Q: np.array = field(default=None) # quadratic state cost
    R: np.array = field(default=None) # quadratic input cost
    Q_f: np.array = field(default=None) # quadratic state cost final
    R_d: np.array = field(default=None) # Quadratic rate cost
    Q_s: float = field(default=1.0) # Quadratic slack cost Q_s*eye
    l_s: float = field(default=1.0) # Linear slack cost l_s*ones

    # named constraints
    v_long_max: float       = field(default = np.inf)
    v_long_min: float       = field(default = -np.inf)
    v_tran_max: float       = field(default = np.inf)
    v_tran_min: float       = field(default = -np.inf)
    psidot_max: float       = field(default = np.inf)
    psidot_min: float       = field(default = -np.inf)
    e_psi_max: float        = field(default = np.inf)
    e_psi_min: float        = field(default = -np.inf)
    s_max: float            = field(default = np.inf)
    s_min: float            = field(default = -np.inf)
    x_tran_max: float       = field(default = np.inf)
    x_tran_min: float       = field(default = -np.inf)
    u_steer_max: float      = field(default = np.inf)
    u_steer_min: float      = field(default = -np.inf)
    u_a_max: float          = field(default = np.inf)
    u_a_min: float          = field(default = -np.inf)
    u_steer_rate_max: float = field(default = np.inf)
    u_steer_rate_min: float = field(default = -np.inf)
    u_a_rate_max: float     = field(default = np.inf)
    u_a_rate_min: float     = field(default = -np.inf)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)


    def __post_init__(self):
        if self.Q is None:
            self.Q = np.ones(self.n)
        if self.R is None:
            self.R = np.ones(self.d)
        if self.Q_f is None:
            self.Q_f = np.zeros(self.n)
        if self.R_d is None:
            self.R_d = np.zeros(self.d)
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self, kinematic=False):
        if kinematic:
            #['v_long', 'e_psi', 's', 'x_tran']  ['u_a', 'u_steer']
            self.state_ub = np.array([self.v_long_max,
                                      self.e_psi_max,
                                      self.s_max,
                                      self.x_tran_max])
            self.state_lb = np.array([self.v_long_min,
                                      self.e_psi_min,
                                      self.s_min,
                                      self.x_tran_min])
        else:
            #['v_long', 'v_tran', 'psidot', 'e_psi', 's', 'x_tran']  ['u_a', 'u_steer']
            self.state_ub = np.array([self.v_long_max,
                                      self.v_tran_max,
                                      self.psidot_max,
                                      self.e_psi_max,
                                      self.s_max,
                                      self.x_tran_max])
            self.state_lb = np.array([self.v_long_min,
                                      self.v_tran_min,
                                      self.psidot_min,
                                      self.e_psi_min,
                                      self.s_min,
                                      self.x_tran_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])
        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min])

        return

@dataclass
class NLLMPCParams(ControllerConfig):
    n: int = field(default=6) # dimension state space
    d: int = field(default=2) # dimension input space

    N: int = field(default=10) # horizon length

    Q: np.array = field(default=None) # quadratic state cost
    R: np.array = field(default=None) # quadratic input cost
    Q_f: np.array = field(default=None) # quadratic state cost final
    R_d: np.array = field(default=None) # Quadratic rate cost
    Q_s: float = field(default=1.0) # Quadratic slack cost
    l_s: float = field(default=1.0) # Linear slack cost
    Q_ch: np.array = field(default=None) # Quadratic cost on convex hull slack

    # named constraints
    v_long_max: float       = field(default = np.inf)
    v_long_min: float       = field(default = -np.inf)
    v_tran_max: float       = field(default = np.inf)
    v_tran_min: float       = field(default = -np.inf)
    psidot_max: float       = field(default = np.inf)
    psidot_min: float       = field(default = -np.inf)
    e_psi_max: float        = field(default = np.inf)
    e_psi_min: float        = field(default = -np.inf)
    s_max: float            = field(default = np.inf)
    s_min: float            = field(default = -np.inf)
    x_tran_max: float       = field(default = 1.0)
    x_tran_min: float       = field(default = -1.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)
    u_a_max: float          = field(default = 2.0)
    u_a_min: float          = field(default = -2.0)
    u_steer_rate_max: float = field(default = 0.5)
    u_steer_rate_min: float = field(default = -0.5)
    u_a_rate_max: float     = field(default = 2.0)
    u_a_rate_min: float     = field(default = -2.0)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    slack: bool = field(default=True)
    solver_dir: str = field(default='')

    n_ss_pts: int = field(default=10) # Number of safe set points per lap
    n_ss_its: int = field(default=3) # Number of previous laps
    ss_selection_weights: np.array = field(default=None)

    safe_set_init_data_file: str = field(default = '')
    safe_set_topic: str          = field(default = 'closed_loop_traj')

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.ones(self.n)
        if self.R is None:
            self.R = np.ones(self.d)
        if self.Q_f is None:
            self.Q_f = np.zeros(self.n)
        if self.R_d is None:
            self.R_d = np.zeros(self.d)
        if self.Q_ch is None:
            self.Q_ch = np.ones(self.n)
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        if self.ss_selection_weights is None:
            self.ss_selection_weights = np.ones(self.n)

    def vectorize_constraints(self, kinematic=False):
        if kinematic:
            #['v_long', 'e_psi', 's', 'x_tran']  ['u_a', 'u_steer']
            self.state_ub = np.array([self.v_long_max,
                                      self.e_psi_max,
                                      self.s_max,
                                      self.x_tran_max])
            self.state_lb = np.array([self.v_long_min,
                                      self.e_psi_min,
                                      self.s_min,
                                      self.x_tran_min])
        else:
            #['v_long', 'v_tran', 'psidot', 'e_psi', 's', 'x_tran']  ['u_a', 'u_steer']
            self.state_ub = np.array([self.v_long_max,
                                      self.v_tran_max,
                                      self.psidot_max,
                                      self.e_psi_max,
                                      self.s_max,
                                      self.x_tran_max])
            self.state_lb = np.array([self.v_long_min,
                                      self.v_tran_min,
                                      self.psidot_min,
                                      self.e_psi_min,
                                      self.s_min,
                                      self.x_tran_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min])
        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min])
        return

@dataclass
class MPCCApproxFullModelParams(ControllerConfig):
    all_tracks: bool = field(default=True)
    n: int = field(default=11) # dimension state space, 1 extra dim for theta
    d: int = field(default=3) # dimension input space, 1 extra dim for v_proj

    N: int = field(default=10)
    Qc: float = field(default=20.0)
    Ql: float = field(default=80.0)
    Q_theta: float = field(default=100.0)
    Q_xref: float = field(default=25.0)
    R_d: float = field(default=0.01)
    R_delta: float = field(default=0.01)

    slack: bool = field(default=False)

    # Slack for collisions
    Q_cs: float = field(default=1.0)  # Quadratic slack cost Q_s*eye
    l_cs: float = field(default=1.0)  # Linear slack cost l_s*ones
    Q_cs_e: float = field(default=1.0) # Quadratic slack cost for safety bound
    l_cs_e: float = field(default=1.0) # Linear slack cost for safety bound
    Q_ts: float = field(default=1.0)  # Quadratic cost for soft track constraint violation
    Q_vmax: float = field(default=1.0)  # Quadratic cost for v_long max soft constraint
    vlong_max_soft: float = field(default=1.0) # Max v_long (only for soft constraint)

    num_std_deviations: float = field(default=1.0)  # Number of std deviations from mean to consider for obstacle slack

    # named constraints
    posx_max: float = field(default=np.inf)
    posx_min: float = field(default=-np.inf)
    posy_max: float = field(default=np.inf)
    posy_min: float = field(default=-np.inf)
    psi_max: float = field(default=np.inf)
    psi_min: float = field(default=-np.inf)
    vx_max: float = field(default=np.inf)
    vx_min: float = field(default=-np.inf)
    vy_max: float = field(default=np.inf)
    vy_min: float = field(default=-np.inf)
    psidot_max: float = field(default=np.inf)
    psidot_min: float = field(default=-np.inf)
    e_psi_max: float = field(default=np.inf)
    e_psi_min: float = field(default=-np.inf)
    s_max: float = field(default=np.inf)
    s_min: float = field(default=-np.inf)
    x_tran_max: float = field(default=np.inf)
    x_tran_min: float = field(default=-np.inf)
    theta_max: float = field(default=np.inf)
    theta_min: float = field(default=0)

    u_a_max: float          = field(default = 2.0)
    u_a_min: float          = field(default = -2.0)
    u_steer_max: float      = field(default = 0.5)
    u_steer_min: float      = field(default = -0.5)
    u_a_rate_max: float     = field(default = 2.0)
    u_a_rate_min: float     = field(default = -2.0)
    u_theta_rate_max: float = field(default = 5.0)
    u_theta_rate_min: float = field(default=-5.0)
    u_steer_rate_max: float     = field(default = 0.5)
    u_steer_rate_min: float     = field(default = -0.5)
    v_proj_max: float = field(default=None)
    v_proj_min: float = field(default=None)
    v_proj_rate_max: float = field(default=np.inf)
    v_proj_rate_min: float = field(default=-np.inf)

    # vector constraints
    state_ub: np.array = field(default=None)
    state_lb: np.array = field(default=None)
    input_ub: np.array = field(default=None)
    input_lb: np.array = field(default=None)
    input_rate_ub: np.array = field(default=None)
    input_rate_lb: np.array = field(default=None)

    optlevel: int = field(default=1)
    solver_dir: str = field(default='')

    def __post_init__(self):
        # TODO Temporary fix
        if self.v_proj_max is None or True:
            self.v_proj_max = self.vx_max*3
        if self.v_proj_min is None or True:
            self.v_proj_min = 0
        if self.state_ub is None:
            self.state_ub = np.inf*np.ones(self.n)
        if self.state_lb is None:
            self.state_lb = -np.inf*np.ones(self.n)
        if self.input_ub is None:
            self.input_ub = np.inf*np.ones(self.d)
        if self.input_lb is None:
            self.input_lb = -np.inf*np.ones(self.d)
        if self.input_rate_ub is None:
            self.input_rate_ub = np.inf*np.ones(self.d)
        if self.input_rate_lb is None:
            self.input_rate_lb = -np.inf*np.ones(self.d)
        self.vectorize_constraints()

    def vectorize_constraints(self):
        self.state_ub = np.array([self.vx_max,
                                  self.vy_max,
                                  self.psidot_max,
                                  self.posx_max,
                                  self.posy_max,
                                  self.psi_max,
                                  self.e_psi_max,
                                  self.s_max,
                                  self.x_tran_max,
                                  self.theta_max])
        self.state_lb = np.array([self.vx_min,
                                  self.vy_min,
                                  self.psidot_min,
                                  self.posx_min,
                                  self.posy_min,
                                  self.psi_min,
                                  self.e_psi_min,
                                  self.s_min,
                                  self.x_tran_min,
                                  self.theta_min])

        self.input_ub = np.array([self.u_a_max, self.u_steer_max, self.v_proj_max])
        self.input_lb = np.array([self.u_a_min, self.u_steer_min, self.v_proj_min])
        self.input_rate_ub = np.array([self.u_a_rate_max, self.u_steer_rate_max, self.v_proj_rate_max])
        self.input_rate_lb = np.array([self.u_a_rate_min, self.u_steer_rate_min, self.v_proj_rate_min])
        return

if __name__ == "__main__":
    pass
