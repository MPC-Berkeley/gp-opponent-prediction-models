from barcgp.common.pytypes import *
from barcgp.controllers.utils.controllerTypes import CAMPCCParams, NLMPCParams
from barcgp.dynamics.models.model_types import DynamicBicycleConfig

# Time discretization
dt = 0.1
# Horizon length
N = 14
# Number of iterations to run PID (need N+1 because of NLMPC predictor warmstart)
n_iter = N+1 
# Track width (should be pre-determined from track generation '.npz')
width = 0.75

# Force rebuild all FORCES code-gen controllers
rebuild = False
# Use a key-point lookahead strategy that is dynamic (all_tracks=True) or pre-generated (all_tracks=False)
all_tracks = True
offset = 32 if not all_tracks else 0

# Initial track conditions
factor = 1.3  # v_long factor

tar_dynamics_config = DynamicBicycleConfig(dt=dt, model_name='dynamic_bicycle',
                                           wheel_dist_front=0.13, wheel_dist_rear=0.13)
ego_dynamics_config = DynamicBicycleConfig(dt=dt, model_name='dynamic_bicycle',
                                           wheel_dist_front=0.13, wheel_dist_rear=0.13)

tarMin = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 0.9, x_tran=-.3 * width, e_psi=-0.02),
                      v=BodyLinearVelocity(v_long=0.5*factor))
tarMax = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 1.2, x_tran=.3* width, e_psi=0.02),
                      v=BodyLinearVelocity(v_long=1.0*factor))
egoMin = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 0.2, x_tran=-.3 * width, e_psi=-0.02),
                      v=BodyLinearVelocity(v_long=0.5*factor))
egoMax = VehicleState(t=0.0,
                      p=ParametricPose(s=offset + 0.4, x_tran=.3 * width, e_psi=0.02),
                      v=BodyLinearVelocity(v_long=1.0*factor))

# Controller parameters
gp_mpcc_ego_params = CAMPCCParams(
    N=14,
    
    qp_interface='hpipm',

    # delay=[1, 1],
    # pos_idx=[3, 4],
    state_scaling=[4.0, 2.0, 7.0, 6.0, 6.0, 6.283185307179586],
    input_scaling=[2.0, 0.436],
    contouring_cost=0.01,
    contouring_cost_N=0.01,
    lag_cost=1000.0,
    lag_cost_N=1000.0,
    performance_cost=0.05,
    vs_cost=0.0001,
    vs_rate_cost=0.001,
    vs_max=5.0,
    vs_min=0.0,
    vs_rate_max=5.0,
    vs_rate_min=-5.0,

    soft_track=True,
    track_slack_quad=30,
    track_slack_lin=250,
    track_tightening=0.1,

    damping=0.25,
    qp_iters=2,
    
    verbose=False
)

mpcc_ego_params = CAMPCCParams(
    N=14,
    
    qp_interface='hpipm',

    # delay=[1, 1],
    # pos_idx=[3, 4],
    state_scaling=[4.0, 2.0, 7.0, 6.0, 6.0, 6.283185307179586],
    input_scaling=[2.0, 0.436],
    contouring_cost=0.01,
    contouring_cost_N=0.01,
    lag_cost=1000.0,
    lag_cost_N=1000.0,
    performance_cost=0.05,
    vs_cost=0.0001,
    vs_rate_cost=0.001,
    vs_max=5.0,
    vs_min=0.0,
    vs_rate_max=5.0,
    vs_rate_min=-5.0,

    soft_track=True,
    track_slack_quad=30,
    track_slack_lin=250,
    track_tightening=0.1,

    damping=0.25,
    qp_iters=2,
    
    verbose=False
)

mpcc_tv_params = CAMPCCParams(
    N=14,
    
    qp_interface='hpipm',

    # delay=[1, 1],
    # pos_idx=[3, 4],
    state_scaling=[4.0, 2.0, 7.0, 6.0, 6.0, 6.283185307179586],
    input_scaling=[2.0, 0.436],
    contouring_cost=0.01,
    contouring_cost_N=0.01,
    lag_cost=1000.0,
    lag_cost_N=1000.0,
    performance_cost=0.05,
    vs_cost=0.0001,
    vs_rate_cost=0.001,
    vs_max=5.0,
    vs_min=0.0,
    vs_rate_max=5.0,
    vs_rate_min=-5.0,

    soft_track=True,
    track_slack_quad=30,
    track_slack_lin=250,
    track_tightening=0.1,

    damping=0.25,
    qp_iters=2,
    
    verbose=False
)


# For NLMPC predictor
nl_mpc_params = NLMPCParams(
        dt=dt,
        solver_dir='' if rebuild else '~/.mpclab_controllers/NL_MPC_solver_forces_pro',
        # solver_dir='',
        optlevel=2,
        slack=False,

        N=N,
        Q=[10.0, 0.2, 1, 15, 0.0, 25.0], # .5 10
        R=[0.1, 0.1],
        Q_f=[10.0, 0.2, 1, 17.0, 0.0, 1.0], # .5 10
        R_d=[5.0, 5.0],
        Q_s=0.0,
        l_s=50.0,

        x_tran_max=width/2,
        x_tran_min=-width/2,
        u_steer_max=0.3,
        u_steer_min=-0.3,
        u_a_max=1.0,
        u_a_min=-2.2,
        u_steer_rate_max=2,
        u_steer_rate_min=-2,
        u_a_rate_max=1.0,
        u_a_rate_min=-1.0
    )
