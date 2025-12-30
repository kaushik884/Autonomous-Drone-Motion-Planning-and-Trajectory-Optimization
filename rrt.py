from ast import List
import gtsam
import numpy as np
import helpers_obstacles as helpers
import drone_dynamics
import cost_function
import math
import boundary_constraints

# TODO 1
def generate_random_point(target: gtsam.Point3) -> gtsam.Point3:
  node = None
  rng = np.random.default_rng(12345)

  if rng.random() < 0.2:
    node = target
  else:
    node = gtsam.Point3(*rng.uniform(0, 10, size=(3,)))

  return node

def distance_euclidean(point1: gtsam.Point3, point2: gtsam.Point3) -> float:

  distance_euclidean = None

  distance_euclidean = np.linalg.norm(point2 - point1)

  return distance_euclidean

# TODO 3
def find_nearest_node(rrt, node):
  nearest_node = None
  index = None

  distances = np.linalg.norm(np.array(rrt) - node, axis=1)
  index = np.argmin(distances)
  nearest_node = rrt[index]

  return nearest_node, index

def steer_naive(parent: gtsam.Point3, target: gtsam.Point3, fraction = 0.2):
  steer_node = None

  displacement = target - parent
  steer_node = parent + displacement * fraction

  return steer_node

def run_rrt(start, target, generate_random_node, steer, distance, find_nearest_node, threshold):
  rrt = []
  parents = []
  max_iterations = 2000
  rrt.append(start)
  parents.append(-1)

  for i in range(max_iterations):
    random_node = generate_random_node(target)
    nearest_node, index = find_nearest_node(rrt, random_node)
    new_node = steer(nearest_node, random_node)
    rrt.append(new_node)
    parents.append(index)
    if (distance(new_node, target) < threshold):
        return rrt, parents

  return rrt, parents

def get_rrt_path(rrt, parents):
  path = []
  i = len(rrt) - 1
  path.append(rrt[i])

  while(parents[i] != -1):
    next = rrt[parents[i]]
    path.append(next)
    i = parents[i]

  path.reverse()
  return path

def drone_racing_rrt(start: gtsam.Pose3, targets):
  drone_path = []
  current = start
  for target in targets:
    rrt_tree, parents = run_rrt(current, target, drone_dynamics.generate_random_pose, drone_dynamics.steer, helpers.distance_between_poses,
                                drone_dynamics.find_nearest_pose, threshold=1.0)
    segment_path = get_rrt_path(rrt_tree, parents)
    helpers.pass_through_the_hoop(target, segment_path)
    drone_path.extend(segment_path[1:])
    current = target

  return drone_path

def run_rrt_with_obstacles(start, target, generate_random_node, steer, distance, find_nearest_node, threshold, obstacles: List = None):

  rrt = []
  parents = []

  max_iterations = 5000
  rrt.append(start)
  parents.append(-1)
  for i in range(max_iterations):
    random_node = generate_random_node(target)
    nearest_node, index = find_nearest_node(rrt, random_node)
    new_node = steer(nearest_node, random_node)
    is_collistion = helpers.check_segment_collision(nearest_node.translation(), new_node.translation(), obstacles)
    if is_collistion:
      continue
    else:
      rrt.append(new_node)
      parents.append(index)
      if (distance(new_node, target) < threshold):
          return rrt, parents

  return rrt, parents

def drone_racing_rrt_with_obstacles(start: gtsam.Pose3, targets, obstacles: List = None):
  drone_path = []
  current = start
  for target in targets:
    rrt_tree, parents = run_rrt_with_obstacles(current, target, drone_dynamics.generate_random_pose, drone_dynamics.steer, helpers.distance_between_poses,
                                               drone_dynamics.find_nearest_pose, 1.0, obstacles)
    path = get_rrt_path(rrt_tree, parents)
    helpers.pass_through_the_hoop(target, path)
    drone_path.extend(path[1:])
    current = target

  return drone_path

# Initialization from RRT Path
def initialize_from_rrt_robust(rrt_path, N: int, dt: float, start_pose: gtsam.Pose3) -> np.ndarray:
    path_length = len(rrt_path)
    states_init = np.zeros((N + 1, 6))

    # Resample RRT path to N+1 knot points
    for i in range(N + 1):
        # Linear interpolation index
        idx_float = i * (path_length - 1) / N
        idx_low = int(np.floor(idx_float))
        idx_high = min(int(np.ceil(idx_float)), path_length - 1)
        alpha = idx_float - idx_low

        # Interpolate position
        pos_low = rrt_path[idx_low].translation()
        if idx_high > idx_low:
            pos_high = rrt_path[idx_high].translation()
            pos = pos_low + alpha * (pos_high - pos_low)
        else:
            pos = pos_low

        # Interpolate attitude (linear on Euler angles with wrapping)
        R_low = rrt_path[idx_low].rotation().matrix()
        yaw_low, pitch_low, roll_low = helpers.euler_from_rotation_matrix_safe(R_low)

        if idx_high > idx_low:
            R_high = rrt_path[idx_high].rotation().matrix()
            yaw_high, pitch_high, roll_high = helpers.euler_from_rotation_matrix_safe(R_high)

            # Handle angle wrapping
            yaw = yaw_low + alpha * cost_function.angle_diff(yaw_high, yaw_low)
            pitch = pitch_low + alpha * cost_function.angle_diff(pitch_high, pitch_low)
            roll = roll_low + alpha * cost_function.angle_diff(roll_high, roll_low)
        else:
            yaw, pitch, roll = yaw_low, pitch_low, roll_low

        states_init[i, :] = [pos[0], pos[1], pos[2], yaw, pitch, roll]

    # Initialize controls: compute from state differences
    controls_init = np.zeros((N, 4))
    for k in range(N):
        # Angle changes
        controls_init[k, 0] = cost_function.angle_diff(states_init[k + 1, 3], states_init[k, 3])  # dyaw
        controls_init[k, 1] = cost_function.angle_diff(states_init[k + 1, 4], states_init[k, 4])  # dpitch
        controls_init[k, 2] = cost_function.angle_diff(states_init[k + 1, 5], states_init[k, 5])  # droll

        # Thrust: start with hover thrust
        controls_init[k, 3] = 10.0  # Newtons

    # Clamp controls to bounds (vectorized)
    deg_to_rad = np.pi / 180
    controls_init[:, 0:3] = np.clip(controls_init[:, 0:3], -10 * deg_to_rad, 10 * deg_to_rad)
    controls_init[:, 3] = np.clip(controls_init[:, 3], 5, 20)

    return cost_function.pack_decision_vars(states_init, controls_init, N)
