import math
import gtsam
import helpers_obstacles as helpers
import rrt 
import drone_dynamics

###########################################################################################
# start_race = gtsam.Pose3(r=gtsam.Rot3.Yaw(math.radians(45)), t=gtsam.Point3(1, 3, 8))
# helpers.drone_racing_path(helpers.get_hoops(), start_race, [])
# helpers.drone_racing_path_with_obstacles(helpers.get_hoops(), start_race, [], obstacles=helpers.get_obstacles_easy())

# start_rrt_3d = gtsam.Point3(1,2,3)
# target_rrt_3d = gtsam.Point3(4,7,2)
# rrt_3d, parents_rrt_3d = rrt.run_rrt(start_rrt_3d, target_rrt_3d, rrt.generate_random_point, rrt.steer_naive, rrt.distance_euclidean, rrt.find_nearest_node, threshold=0.1)
# print("Nodes in RRT: ", len(rrt_3d))
# helpers.visualize_tree(rrt_3d, parents_rrt_3d, start_rrt_3d, target_rrt_3d)

###########################################################################################
# path_rrt_3d = rrt.get_rrt_path(rrt_3d, parents_rrt_3d)
# print("Length of Path: ", len(path_rrt_3d))
# helpers.visualize_path(path_rrt_3d, start_rrt_3d, target_rrt_3d)

# start_rrt_drone = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 2, 3))
# target_rrt_drone = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(5, 7, 4))
# rrt_drone, parents_rrt_drone = rrt.run_rrt(start_rrt_drone, target_rrt_drone, drone_dynamics.generate_random_pose, drone_dynamics.steer_with_terminal_velocity,
#                                        helpers.distance_between_poses, drone_dynamics.find_nearest_pose, threshold=1.5)
# print("Number of RRT Nodes: ", len(rrt_drone))
# helpers.visualize_tree(rrt_drone, parents_rrt_drone, start_rrt_drone, target_rrt_drone)

###########################################################################################
# path_rrt_drone = rrt.get_rrt_path(rrt_drone, parents_rrt_drone)
# print("Length of Path: ", len(path_rrt_drone))
# print("The path obtained by steering with a terminal velocity:")
# helpers.animate_drone_path(path_rrt_drone, start_rrt_drone, target_rrt_drone)

# start_rrt_drone_realistic = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 2, 3))
# target_rrt_drone_realistic = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(8, 5, 6))
# rrt_drone_realistic, parents_rrt_drone_realistic = rrt.run_rrt(start_rrt_drone_realistic, target_rrt_drone_realistic,
#                                                            drone_dynamics.generate_random_pose, drone_dynamics.steer, helpers.distance_between_poses,
#                                                            drone_dynamics.find_nearest_pose, threshold=1.5)
# print("Nodes in RRT Tree: ", len(rrt_drone_realistic))
# helpers.visualize_tree(rrt_drone_realistic, parents_rrt_drone_realistic, start_rrt_drone_realistic, target_rrt_drone_realistic)

###########################################################################################
# path_rrt_drone_realistic = rrt.get_rrt_path(rrt_drone_realistic, parents_rrt_drone_realistic)
# print("Length of Path: ", len(path_rrt_drone_realistic))
# helpers.animate_drone_path(path_rrt_drone_realistic, start_rrt_drone_realistic, target_rrt_drone_realistic)

###########################################################################################
# start_rrt_drone_race = gtsam.Pose3(r=gtsam.Rot3(), t=gtsam.Point3(1, 3, 8))
# targets_rrt_drone_race = helpers.get_targets()
# path_rrt_drone_race = rrt.drone_racing_rrt(start_rrt_drone_race, targets_rrt_drone_race)
# helpers.drone_racing_path(helpers.get_hoops(), start_rrt_drone_race, path_rrt_drone_race)
# print("Length of Drone Racing Path: ", len(path_rrt_drone_race))

###########################################################################################
# Test RRT with obstacles on simple scenario
# print("Testing RRT with obstacle avoidance...")

# # Simple test: navigate around a single sphere obstacle
# start_obs = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(2, 2, 5))
# target_obs = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(8, 8, 8))

# # Create obstacle in the middle
# obstacles_simple = [
#     helpers.SphereObstacle(center=[5, 5, 6.5], radius=1.5, name="Central Pillar")
# ]

# # Run RRT with obstacles
# rrt_obs, parents_obs = rrt.run_rrt_with_obstacles(
#     start=start_obs,
#     target=target_obs,
#     generate_random_node=drone_dynamics.generate_random_pose,
#     steer=drone_dynamics.steer,
#     distance=helpers.distance_between_poses,
#     find_nearest_node=drone_dynamics.find_nearest_pose,
#     threshold=1.5,
#     obstacles=obstacles_simple
# )

# print(f"RRT with obstacles completed: {len(rrt_obs)} nodes")

# # Extract and verify path
# path_obs = rrt.get_rrt_path(rrt_obs, parents_obs)
# print(f"Path length: {len(path_obs)} waypoints")

# # Verify no collisions
# has_collision, _ = helpers.check_path_collision(path_obs, obstacles_simple)
# if not has_collision:
#     print(f"RRT is Collision-free!")
# else:
#     print(f"RRT may have collisions")

# # Visualize RRT with obstacles
# helpers.visualize_path_with_obstacles(
#     path_obs, start_obs, target_obs, obstacles_simple
# )

###########################################################################################
# Run drone racing with obstacles!
# print("Starting drone racing with obstacles...")

# start_racing_obs = gtsam.Pose3(r=gtsam.Rot3(), t=gtsam.Point3(1, 3, 8))
# targets_racing_obs = helpers.get_targets()

# # Get obstacle configuration (easy difficulty)
# obstacles_racing = helpers.get_obstacles_easy()
# print(f"Racing with {len(obstacles_racing)} obstacles")

# # Execute racing
# path_racing_obs = rrt.drone_racing_rrt_with_obstacles(
#     start_racing_obs,
#     targets_racing_obs,
#     obstacles_racing
# )

# # Visualize racing path with obstacles
# helpers.drone_racing_path_with_obstacles(
#     hoops=helpers.get_hoops(),
#     start=start_racing_obs,
#     path=path_racing_obs,
#     obstacles=obstacles_racing
# )

# print(f"\nTotal racing path length: {len(path_racing_obs)} waypoints")

# # Verify collision-free
# has_collision, _ = helpers.check_path_collision(path_racing_obs, obstacles_racing)
# if not has_collision:
#     print(f"RRT is Collision-free!")
# else:
#     print(f"RRT may have collisions")

###########################################################################################
