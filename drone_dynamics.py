import gtsam
import math
import numpy as np
import helpers_obstacles as helpers

def compute_attitude_from_ypr(yaw: float, pitch: float, roll: float) -> gtsam.Rot3:
  attitude = None
  attitude = gtsam.Rot3.Ypr(yaw, pitch, roll)

  return attitude

def compute_force(attitude: gtsam.Rot3, thrust: float) -> gtsam.Point3:
  force = None
  force = attitude.rotate(gtsam.Point3(0, 0, thrust))

  return force

def compute_terminal_velocity(force: gtsam.Point3, kd: float = 0.0425) -> gtsam.Point3:
  terminal_velocity = None
  eps = 1e-6

  force_array = np.array([force[0], force[1], force[2]])
  terminal_velocity_array = np.sign(force_array) * np.sqrt((np.abs(force_array) + eps) / kd)
  terminal_velocity = gtsam.Point3(*terminal_velocity_array)

  return terminal_velocity

def generate_random_pose(target: gtsam.Pose3) -> gtsam.Pose3:
  rng = np.random.default_rng(12345)
  node = None
  if rng.random() < 0.2:
    node = target
  else:
    position = gtsam.Point3(*rng.uniform(0, 10, size=(3,)))
    attitude = compute_attitude_from_ypr(rng.uniform(-np.pi/3, np.pi/3), rng.uniform(-np.pi/3, np.pi/3), rng.uniform(-np.pi/3, np.pi/3))
    node = gtsam.Pose3(attitude, position)

  return node

def find_nearest_pose(rrt, node):
  nearest = None
  index = None

  distances = []
  for i in range(len(rrt)):
    distances.append(helpers.distance_between_poses(rrt[i], node))

  index = np.argmin(distances)
  nearest = rrt[index]
  return nearest, index

def steer_with_terminal_velocity(current: gtsam.Pose3, target: gtsam.Pose3, duration: float = 0.1) -> gtsam.Pose3:
  steer_node = None

  direction = target.translation() - current.translation()

  direction_np = np.array([direction[0], direction[1], direction[2]])
  direction_norm = direction_np / np.linalg.norm(direction_np)
  direction_normalized = gtsam.Point3(direction_norm[0], direction_norm[1], direction_norm[2])

  attitude = helpers.get_new_attitude(current, direction_normalized)

  force = compute_force(attitude, 20.0)

  terminal_velocity = compute_terminal_velocity(force)

  node = current.translation() + duration * terminal_velocity

  steer_node = gtsam.Pose3(attitude, node)

  return steer_node

def compute_force_with_gravity(attitude: gtsam.Rot3, thrust: float, mass: float = 1.0) -> gtsam.Point3:
  force = None
  g = 10.0  # m/s^2

  weight = gtsam.Point3(0, 0, -mass*g)
  T_world = attitude.rotate(gtsam.Point3(0, 0, thrust))
  force = T_world + weight
  
  return force

def steer(current: gtsam.Pose3, target: gtsam.Pose3, duration = 0.1):
  steer_node = None
  yaw_values = [-10, 0, 10]
  pitch_values = [-10, 0, 10]
  roll_values = [-10, 0, 10]
  thrust_values = [5, 10, 15, 20]

  best_node = None
  best_distance = float('inf')

  for yaw in yaw_values:
      for pitch in pitch_values:
          for roll in roll_values:
              for thrust in thrust_values:
                  curr_attitude = compute_attitude_from_ypr(np.radians(yaw), np.radians(pitch), np.radians(roll))

                  attitude = current.rotation() * curr_attitude

                  force = compute_force_with_gravity(attitude, thrust)

                  terminal_velocity = compute_terminal_velocity(force)

                  position = current.translation() + duration * terminal_velocity

                  node = gtsam.Pose3(attitude, position)

                  distance = helpers.distance_between_poses(node, target)

                  if distance < best_distance:
                      best_distance = distance
                      best_node = node

  steer_node = best_node

  return steer_node
