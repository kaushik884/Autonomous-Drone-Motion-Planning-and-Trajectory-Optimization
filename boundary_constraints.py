import numpy as np
import drone_dynamics
import cost_function
import gtsam
import helpers_obstacles as helpers

def dynamics_constraints_robust(z: np.ndarray, N: int, dt: float) -> np.ndarray:
    violations = []
    mass = 1.0  # kg
    g = 10.0    # m/s^2 (gravity)

    states, controls = cost_function.unpack_decision_vars(z, N)

    for k in range(N):
        p_k = states[k, :3]
        yaw_k = states[k, 3]
        pitch_k = states[k, 4]
        roll_k = states[k, 5]

        p_k1 = states[k+1, :3]
        yaw_k1 = states[k+1, 3]
        pitch_k1 = states[k+1, 4]
        roll_k1 = states[k+1, 5]

        d_yaw = controls[k, 0]
        d_pitch = controls[k, 1]
        d_roll = controls[k, 2]
        thrust_k = controls[k, 3]

        attitude = drone_dynamics.compute_attitude_from_ypr(yaw_k, pitch_k, roll_k)
        force = drone_dynamics.compute_force_with_gravity(attitude, thrust_k, mass)

        terminal_velocity = drone_dynamics.compute_terminal_velocity(force, kd=0.0425)

        # predicted p_k+1
        p_k1_predicted = p_k + terminal_velocity * dt
        position_violation = p_k1 - p_k1_predicted # "If I apply thrust T with attitude (ψ,θ,φ), do I end up at p_{k+1}?" this is implemented here

        # predicted yaw, pitch, roll values
        yaw_k1_predicted = yaw_k + d_yaw
        pitch_k1_predicted = pitch_k + d_pitch
        roll_k1_predicted = roll_k + d_roll

        # "If I apply angular control Δη, do I get attitude η_{k+1}?" this is implemented here
        yaw_violation = cost_function.angle_diff(yaw_k1, yaw_k1_predicted)
        pitch_violation = cost_function.angle_diff(pitch_k1, pitch_k1_predicted)
        roll_violation = cost_function.angle_diff(roll_k1, roll_k1_predicted)

        violations.extend(position_violation)
        violations.extend([yaw_violation, pitch_violation, roll_violation])

    return np.array(violations)

def boundary_constraints_robust(z: np.ndarray, N: int, start_pose: gtsam.Pose3, goal_position: np.ndarray, hoops) -> np.ndarray:

    violations = []

    states, controls = cost_function.unpack_decision_vars(z, N)

    p_start = states[0, :3]
    yaw_start = states[0, 3]
    pitch_start = states[0, 4]
    roll_start = states[0, 5]

    # get the required start position and orientation
    desired_p_start = start_pose.translation()
    desired_p_start_array = np.array([desired_p_start[0], desired_p_start[1], desired_p_start[2]])

    # find euler angles from start_pose rotation matrix
    R_start = start_pose.rotation().matrix()
    desired_yaw, desired_pitch, desired_roll = helpers.euler_from_rotation_matrix_safe(R_start)

    # position violations
    position_start_violation = p_start - desired_p_start_array

    # orientation violations
    yaw_start_violation = cost_function.angle_diff(yaw_start, desired_yaw)
    pitch_start_violation = cost_function.angle_diff(pitch_start, desired_pitch)
    roll_start_violation = cost_function.angle_diff(roll_start, desired_roll)

    violations.extend(position_start_violation)
    violations.extend([yaw_start_violation, pitch_start_violation, roll_start_violation])

    p_goal = states[N, :3]

    # goal position violation
    position_goal_violation = p_goal - goal_position
    violations.extend(position_goal_violation)

    if len(hoops) > 0:
        positions = states[:, :3]

        hoop_positions = [hoop.translation() for hoop in hoops]
        hoop_positions_array = [np.array([hp[0], hp[1], hp[2]]) for hp in hoop_positions]

        hoop_indices = helpers.find_hoop_indices_robust(positions, hoop_positions_array, N)

        for h, hoop_idx in enumerate(hoop_indices):
            p_at_hoop = states[hoop_idx, :3]

            hoop_center = hoops[h].translation()
            hoop_center_array = np.array([hoop_center[0], hoop_center[1], hoop_center[2]])

            hoop_violation = p_at_hoop - hoop_center_array
            violations.extend(hoop_violation)

    return np.array(violations)

def collision_constraints_optimized(z: np.ndarray, N: int, obstacles, subsample: int = 2) -> np.ndarray:
    states, controls = cost_function.unpack_decision_vars(z, N)
    violations = []
    safety_margin = 0.5

    for k in range(0, N + 1, subsample):
        px, py, pz = states[k, 0:3]
        # point = gtsam.Point3(px, py, pz)

        for obs in obstacles:
            if isinstance(obs, helpers.SphereObstacle):
                # Vectorized distance computation
                dist = np.linalg.norm(np.array([px, py, pz]) - np.array(obs.center))
                violations.append(obs.radius + safety_margin - dist)

    return np.array(violations)
