from barcgp.common.pytypes import *
from barcgp.controllers.utils.controllerTypes import CAMPCCParams, NLMPCParams
from barcgp.dynamics.models.model_types import DynamicBicycleConfig

# Time discretization for control
dt = 0.1
# Time discretization for simulation
sim_dt = 0.01

# Horizon length
N = 14
# Number of iterations to run PID (need N+1 because of NLMPC predictor warmstart)
n_iter = N+1 
# Track width (should be pre-determined from track generation '.npz')
width = 1.1

# Force rebuild all FORCES code-gen controllers
rebuild = False
# Use a key-point lookahead strategy that is dynamic (all_tracks=True) or pre-generated (all_tracks=False)
all_tracks = True
offset = 32 if not all_tracks else 0

# Velocity limits
ego_v_max = 3.0
tar_v_max = 2.5

# Initial track conditions
# factor = 1.3  # v_long factor
factor = 3.0

# Cost weights
tar_blocking_weight = 20
tar_contouring_nominal = 0.1
ego_inner_collision_quad = 50
ego_outer_collision_quad = 5

# Cost parameters
ego_inner_radius = 0.2
ego_outer_radius = 0.3
tar_radius = 0.22

ego_L = 0.37
ego_W = 0.195

tar_L = 0.37
tar_W = 0.195

# Dynamics configs
discretization_method = 'rk4'

ego_dynamics_config = DynamicBicycleConfig(dt=dt,
                                            model_name='dynamic_bicycle',
                                            noise=False,
                                            discretization_method=discretization_method,
                                            simple_slip=False,
                                            tire_model='pacejka',
                                            mass=2.2187,
                                            yaw_inertia=0.02723,
                                            wheel_friction=0.9,
                                            pacejka_b_front=5.0,
                                            pacejka_b_rear=5.0,
                                            pacejka_c_front=2.28,
                                            pacejka_c_rear=2.28)

ego_sim_dynamics_config = DynamicBicycleConfig(dt=sim_dt,
                                            model_name='dynamic_bicycle',
                                            noise=False,
                                            discretization_method=discretization_method,
                                            simple_slip=False,
                                            tire_model='pacejka',
                                            mass=2.2187,
                                            yaw_inertia=0.02723,
                                            wheel_friction=0.9,
                                            pacejka_b_front=5.0,
                                            pacejka_b_rear=5.0,
                                            pacejka_c_front=2.28,
                                            pacejka_c_rear=2.28)

tar_dynamics_config = DynamicBicycleConfig(dt=dt,
                                            model_name='dynamic_bicycle',
                                            noise=False,
                                            discretization_method=discretization_method,
                                            simple_slip=False,
                                            tire_model='pacejka',
                                            mass=2.2187,
                                            yaw_inertia=0.02723,
                                            wheel_friction=0.9,
                                            pacejka_b_front=5.0,
                                            pacejka_b_rear=5.0,
                                            pacejka_c_front=2.28,
                                            pacejka_c_rear=2.28)

tar_sim_dynamics_config = DynamicBicycleConfig(dt=0.01,
                                            model_name='dynamic_bicycle',
                                            noise=False,
                                            discretization_method=discretization_method,
                                            simple_slip=False,
                                            tire_model='pacejka',
                                            mass=2.2187,
                                            yaw_inertia=0.02723,
                                            wheel_friction=0.9,
                                            pacejka_b_front=5.0,
                                            pacejka_b_rear=5.0,
                                            pacejka_c_front=2.28,
                                            pacejka_c_rear=2.28)

# Initial condition sampling bounds
ep = 5*np.pi/180
tarMin = VehicleState(t=0.0,
                    p=ParametricPose(s=offset + 0.9, x_tran=-.9 * width/2, e_psi=-ep),
                    v=BodyLinearVelocity(v_long=0.5*tar_v_max))
tarMax = VehicleState(t=0.0,
                    p=ParametricPose(s=offset + 1.2, x_tran=.9 * width/2, e_psi=ep),
                    v=BodyLinearVelocity(v_long=1.0*tar_v_max))
egoMin = VehicleState(t=0.0,
                    p=ParametricPose(s=offset + 0.2, x_tran=-.9 * width/2, e_psi=-ep),
                    v=BodyLinearVelocity(v_long=0.5*ego_v_max))
egoMax = VehicleState(t=0.0,
                    p=ParametricPose(s=offset + 0.4, x_tran=.9 * width/2, e_psi=ep),
                    v=BodyLinearVelocity(v_long=1.0*ego_v_max))

ego_state_input_ub=VehicleState(x=Position(x=100, y=100),
                            e=OrientationEuler(psi=100),
                            v=BodyLinearVelocity(v_long=ego_v_max, v_tran=2.0),
                            w=BodyAngularVelocity(w_psi=100),
                            u=VehicleActuation(u_a=2.0, u_steer=0.436))
ego_state_input_lb=VehicleState(x=Position(x=-100, y=-100),
                            e=OrientationEuler(psi=-100),
                            v=BodyLinearVelocity(v_long=-1.0, v_tran=-2.0),
                            w=BodyAngularVelocity(w_psi=-100),
                            u=VehicleActuation(u_a=-2.0, u_steer=-0.436))
ego_input_rate_ub = VehicleState(u=VehicleActuation(u_a=10.0, u_steer=4.5))
ego_input_rate_lb = VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-4.5))

tar_state_input_ub=VehicleState(x=Position(x=100, y=100),
                            e=OrientationEuler(psi=100),
                            v=BodyLinearVelocity(v_long=tar_v_max, v_tran=2.0),
                            w=BodyAngularVelocity(w_psi=100),
                            u=VehicleActuation(u_a=2.0, u_steer=0.436))
tar_state_input_lb=VehicleState(x=Position(x=-100, y=-100),
                            e=OrientationEuler(psi=-100),
                            v=BodyLinearVelocity(v_long=-1.0, v_tran=-2.0),
                            w=BodyAngularVelocity(w_psi=-100),
                            u=VehicleActuation(u_a=-2.0, u_steer=-0.436))
tar_input_rate_ub = VehicleState(u=VehicleActuation(u_a=10.0, u_steer=4.5))
tar_input_rate_lb = VehicleState(u=VehicleActuation(u_a=-10.0, u_steer=-4.5))

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

ego_mpc_params = CAMPCCParams(dt=dt, N=N,
                            verbose=False,
                            debug_plot=False,
                            damping=0.25,
                            qp_iters=2,
                            state_scaling=[4, 2, 7, 6, 6, 2*np.pi],
                            input_scaling=[2, 0.436],
                            delay=None,
                            contouring_cost=0.01,
                            contouring_cost_N=0.01,
                            lag_cost=1000.0,
                            lag_cost_N=1000.0,
                            performance_cost=0.05,
                            vs_cost=1e-4,
                            vs_rate_cost=1e-3,
                            vs_max=5.0,
                            vs_min=0.0,
                            vs_rate_max=5.0,
                            vs_rate_min=-5.0,
                            soft_track=True,
                            track_slack_quad=50.0,
                            track_slack_lin=250.0,
                            track_tightening=0.1,
                            code_gen=False,
                            opt_flag='O3',
                            solver_name='ego_MPCC_conv',
                            qp_interface='hpipm')

tar_mpc_params = CAMPCCParams(dt=dt, N=N,
                            verbose=False,
                            debug_plot=False,
                            damping=0.25,
                            qp_iters=2,
                            state_scaling=[4, 2, 7, 6, 6, 2*np.pi],
                            input_scaling=[2, 0.436],
                            delay=None,
                            parametric_contouring_cost=True,
                            # contouring_cost=0.01,
                            # contouring_cost_N=0.01,
                            lag_cost=1000.0,
                            lag_cost_N=1000.0,
                            performance_cost=0.02,
                            vs_cost=1e-4,
                            vs_rate_cost=1e-3,
                            vs_max=5.0,
                            vs_min=0.0,
                            vs_rate_max=5.0,
                            vs_rate_min=-5.0,
                            soft_track=True,
                            track_slack_quad=50.0,
                            track_slack_lin=250.0,
                            track_tightening=0.1,
                            code_gen=False,
                            opt_flag='O3',
                            solver_name='tar_MPCC_conv',
                            qp_interface='hpipm')

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

pred_mpc_params = CAMPCCParams(dt=dt, N=N,
                            verbose=False,
                            debug_plot=False,
                            damping=0.25,
                            qp_iters=2,
                            state_scaling=[4, 2, 7, 6, 6, 2*np.pi],
                            input_scaling=[2, 0.436],
                            delay=None,
                            parametric_contouring_cost=False,
                            contouring_cost=tar_contouring_nominal,
                            contouring_cost_N=tar_contouring_nominal,
                            lag_cost=1000.0,
                            lag_cost_N=1000.0,
                            performance_cost=0.02,
                            vs_cost=1e-4,
                            vs_rate_cost=1e-3,
                            vs_max=5.0,
                            vs_min=0.0,
                            vs_rate_max=5.0,
                            vs_rate_min=-5.0,
                            soft_track=True,
                            track_slack_quad=50.0,
                            track_slack_lin=250.0,
                            track_tightening=0.1,
                            code_gen=False,
                            opt_flag='O3',
                            solver_name='pred_MPCC_conv',
                            qp_interface='hpipm')