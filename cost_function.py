import numpy as np

def angle_diff(angle_to: float, angle_from: float) -> float:

    diff = np.arctan2(np.sin(angle_to - angle_from), np.cos(angle_to - angle_from))

    return diff

def pack_decision_vars(states: np.ndarray, controls: np.ndarray, N: int) -> np.ndarray:

    z = np.concatenate((states.flatten(), controls.flatten()))

    return z

def unpack_decision_vars(z: np.ndarray, N: int):
    states_size = (N + 1) * 6

    states_flat = z[:states_size]
    controls_flat = z[states_size:]

    states = states_flat.reshape((N + 1, 6))
    controls = controls_flat.reshape((N, 4))

    return states, controls

def cost_function_thrust(states: np.ndarray, controls: np.ndarray, N: int, weight: float = 0.1) -> float:
    T_hover = 10.0
    cost = 0.0
    deviations = controls[:, 3] - T_hover
    cost = weight * np.sum(deviations**2)
    
    return cost

def cost_function_angular(states: np.ndarray, controls: np.ndarray, N: int, weight: float = 1.0) -> float:
    cost = 0.0
    velo = controls[:, :3]
    cost = weight * np.sum(velo**2)

    return cost

def cost_function_smoothness(states: np.ndarray, controls: np.ndarray, N: int, weight: float = 5.0) -> float:
    if N <= 1:
        return 0.0

    cost = 0.0
    cost = weight * np.sum((controls[1:, :] - controls[:-1, :])**2)

    return cost

def cost_function_gimbal_lock(states: np.ndarray, controls: np.ndarray, N: int) -> float:
    cost = 0.0
    pitch = np.abs(states[:, 4])
    limit = 5 * np.pi / 18
    cost = 1000 * np.sum(np.maximum(0, pitch - limit)**2)

    return cost

def cost_function_tuned(z: np.ndarray, N: int, weights) -> float:
    states, controls = unpack_decision_vars(z, N)

    cost = 0.0

    # Add thrust cost
    cost += cost_function_thrust(states, controls, N, weights['thrust'])

    # Add angular velocity cost
    cost += cost_function_angular(states, controls, N, weights['angular'])

    # Add smoothness cost
    cost += cost_function_smoothness(states, controls, N, weights['smoothness'])

    # Add gimbal lock penalty
    cost += cost_function_gimbal_lock(states, controls, N)

    return cost