import gtsam
import numpy as np
import helpers_obstacles as helpers
import rrt
import boundary_constraints
import cost_function
import time

###########################################################################################
# # Define simple scenario
# start_simple = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(2, 2, 5))
# goal_simple = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(8, 8, 5))

# print("Step 1: Running RRT...")
# rrt_simple, parents_simple = rrt.run_rrt(
#     start_simple, goal_simple,
#     rrt.drone_dynamics.generate_random_pose, rrt.drone_dynamics.steer, helpers.distance_between_poses,
#     rrt.drone_dynamics.find_nearest_pose, threshold=2.0
# )
# path_simple = rrt.get_rrt_path(rrt_simple, parents_simple)
# print(f"RRT path: {len(path_simple)} waypoints")

# print("\nStep 2: Optimizing trajectory...")
# optimized_simple, success_simple, info_simple = helpers.optimize_trajectory(
#     rrt_path=path_simple,
#     start_pose=start_simple,
#     goal_position=goal_simple.translation(),
#     hoops=[],
#     obstacles=[],
#     N=20,
#     dt=0.1,
#     weights={'thrust': 0.1, 'angular': 1.0, 'smoothness': 5.0},
#     # Pass student-implemented functions (TODOs 17-28)
#     initialize_from_rrt_robust=rrt.initialize_from_rrt_robust,
#     dynamics_constraints_robust=boundary_constraints.dynamics_constraints_robust,
#     boundary_constraints_robust=boundary_constraints.boundary_constraints_robust,
#     collision_constraints_optimized=boundary_constraints.collision_constraints_optimized,
#     cost_function_tuned=cost_function.cost_function_tuned,
#     unpack_decision_vars=cost_function.unpack_decision_vars
# )

# if success_simple:
#     print(f"\n✅ SUCCESS! Cost: {info_simple['cost']:.2f}, Iterations: {info_simple['iterations']}")
# else:
#     print(f"\n❌ Optimization failed, using RRT path")

# print("\nStep 3: Visualizing...")

# # Visualize comparison
# fig = helpers.visualize_rrt_vs_optimized_comparison(
#     path_simple, optimized_simple, start_simple, goal_simple,
#     title="Demo 1: RRT vs Optimized"
# )
# fig.show()

###########################################################################################
# Get racing setup
# hoops_demo2 = helpers.get_hoops()
# targets_demo2 = helpers.get_targets()
# start_demo2 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 3, 8))

# print("Step 1: Running RRT through hoops...")
# rrt_race_path = rrt.drone_racing_rrt(start_demo2, targets_demo2)
# print(f"RRT path: {len(rrt_race_path)} waypoints")

# print("\nStep 2: Optimizing racing trajectory...")
# optimized_race, success_race, info_race = helpers.optimize_racing_path_sequential(
#     rrt_path=rrt_race_path,
#     start_pose=start_demo2,
#     hoops=hoops_demo2,
#     obstacles=[],
#     N=25,
#     dt=0.1,
#     weights={'thrust': 0.05, 'angular': 0.5, 'smoothness': 5.0},
#     initialize_from_rrt_robust=rrt.initialize_from_rrt_robust,
#     dynamics_constraints_robust=boundary_constraints.dynamics_constraints_robust,
#     boundary_constraints_robust=boundary_constraints.boundary_constraints_robust,
#     collision_constraints_optimized=boundary_constraints.collision_constraints_optimized,
#     cost_function_tuned=cost_function.cost_function_tuned,
#     unpack_decision_vars=cost_function.unpack_decision_vars
# )

# if success_race:
#     print(f"\n✅ Racing optimization SUCCESS!")
# else:
#     print(f"\n⚠ Some segments may have failed")


# print("\nStep 3: Visualizing...")

# # Visualize racing comparison
# fig = helpers.drone_racing_path_comparison(
#     hoops_demo2, start_demo2, rrt_race_path, optimized_race,
#     title="Demo 2: RRT vs Optimized Racing"
# )
# fig.show()

###########################################################################################
print("\n" + "="*80)
print("🏁 FINAL CHALLENGE: RACING WITH OBSTACLES (EASY)")
print("="*80 + "\n")

# Setup
hoops_final = helpers.get_hoops()
targets_final = helpers.get_targets()
start_final = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 3, 8))
obstacles_easy = helpers.get_obstacles_easy()

print(f"Configuration:")
print(f"   - Start: {start_final.translation()}")
print(f"   - Hoops: {len(hoops_final)}")
print(f"   - Obstacles: {len(obstacles_easy)} (EASY)")

print("\n" + "-"*80)
print("STAGE 1: RRT WITH OBSTACLES")
print("-"*80)

import time
start_time = time.time()
rrt_final_path = rrt.drone_racing_rrt_with_obstacles(start_final, targets_final, obstacles_easy)
rrt_time = time.time() - start_time

print(f"RRT completed in {rrt_time:.2f}s")
print(f"Path: {len(rrt_final_path)} waypoints")

has_collision, _ = helpers.check_path_collision(rrt_final_path, obstacles_easy)
if not has_collision:
    print(f"RRT is Collision-free!")
else:
    print(f"RRT may have collisions")

# Visualize RRT racing path with obstacles once again
helpers.drone_racing_path_with_obstacles(
    hoops=helpers.get_hoops(),
    start=start_final,
    path=rrt_final_path,
    obstacles=obstacles_easy
)

print(f"\nTotal racing path length: {len(rrt_final_path)} waypoints")

# Verify collision-free
has_collision, _ = helpers.check_path_collision(rrt_final_path, obstacles_easy)
if not has_collision:
    print(f"RRT is Collision-free!")
else:
    print(f"RRT may have collisions")

print("\n" + "-"*80)
print("STAGE 2: TRAJECTORY OPTIMIZATION")
print("-"*80)

opt_start = time.time()
optimized_final, success_final, info_final = helpers.optimize_racing_path_sequential(
    rrt_path=rrt_final_path,
    start_pose=start_final,
    hoops=hoops_final,
    obstacles=obstacles_easy,
    N=25,
    dt=0.1,
    weights={'thrust': 0.05, 'angular': 0.5, 'smoothness': 5.0},
    initialize_from_rrt_robust=rrt.initialize_from_rrt_robust,
    dynamics_constraints_robust=boundary_constraints.dynamics_constraints_robust,
    boundary_constraints_robust=boundary_constraints.boundary_constraints_robust,
    collision_constraints_optimized=boundary_constraints.collision_constraints_optimized,
    cost_function_tuned=cost_function.cost_function_tuned,
    unpack_decision_vars=cost_function.unpack_decision_vars
)
opt_time = time.time() - opt_start

if success_final:
    print(f"\n✅ OPTIMIZATION SUCCESS!")
    print(f"   Time: {opt_time:.2f}s")
    print(f"   Total time: {rrt_time + opt_time:.2f}s")

    has_collision_opt, _ = helpers.check_path_collision(optimized_final, obstacles_easy)
    if not has_collision_opt:
        print(f"Optimized path is collision-free!")
    print(f"Passes through all {len(hoops_final)} hoops")
else:
    print(f"\n Optimization had issues, using RRT path")
    optimized_final = rrt_final_path

print("\n" + "-"*80)
print("STAGE 3: VISUALIZATION")
print("-"*80)

helpers.drone_racing_path_with_obstacles(
    hoops_final, start_final, optimized_final, obstacles_easy,
)

print("\n✨ Interactive 3D visualization above!")
print("   - Rotate: mouse drag")
print("   - Zoom: scroll wheel")
print("   - Pan: right-click drag")


def compute_path_length(path):
    """Compute total path length"""
    length = 0.0
    for i in range(len(path)-1):
        pos1 = np.array(path[i].translation())
        pos2 = np.array(path[i+1].translation())
        length += np.linalg.norm(pos2 - pos1)
    return length

# Path Metric Evaluation:
length_rrt_final = compute_path_length(rrt_final_path)
length_opt_final = compute_path_length(optimized_final)

print("PATH COMPARISON:")
print(f"  RRT path length: {length_rrt_final:.2f} m")
print(f"  Optimized path length: {length_opt_final:.2f} m")
print(f"  Reduction: {(length_rrt_final-length_opt_final)/length_rrt_final*100:.1f}%")
print(f"\n  RRT waypoints: {len(rrt_final_path)}")
print(f"  Optimized waypoints: {len(optimized_final)}")
if success_final:
    print(f"\n✨ The optimized path is smoother and more efficient!")

###########################################################################################

# # Setup HARD course
# hoops_hard = helpers.get_hoops()
# targets_hard = helpers.get_targets()
# start_hard = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 3, 8))
# obstacles_easy = helpers.get_obstacles_easy()
# obstacles_hard = helpers.get_obstacles_hard()

# print(f"   - Hoops: {len(hoops_hard)}")
# print(f"   - Obstacles: {len(obstacles_hard)} (vs {len(obstacles_easy)} in easy)")
# print(f"   - Expected time: 30-90s")

# # You may need to tweak RRT parameters for success here

# print("\n" + "-"*80)
# print("STAGE 1: RRT WITH HARD OBSTACLES")
# print("-"*80)

# # Try RRT multiple times
# rrt_hard_success = False
# max_attempts = 3

# for attempt in range(1, max_attempts + 1):
#     print(f"\nAttempt {attempt}/{max_attempts}...")
#     try:
#         start_time_hard = time.time()
#         rrt_hard_path = rrt.drone_racing_rrt_with_obstacles(start_hard, targets_hard, obstacles_hard)
#         rrt_hard_time = time.time() - start_time_hard

#         has_collision_rrt, _ = helpers.check_path_collision(rrt_hard_path, obstacles_hard)

#         if not has_collision_rrt and len(rrt_hard_path) > 0:
#             print(f"✓ SUCCESS on attempt {attempt}!")
#             print(f"✓ Time: {rrt_hard_time:.2f}s")
#             print(f"✓ Path: {len(rrt_hard_path)} waypoints")
#             rrt_hard_success = True
#             break
#         else:
#             print(f"❌ Attempt {attempt} failed")
#     except Exception as e:
#         print(f"❌ Error: {e}")

# if not rrt_hard_success:
#     print(f"\n⚠️ RRT failed after {max_attempts} attempts")
#     print(f"   Suggestions:")
#     print(f"   - Increase RRT threshold")
#     print(f"   - Add goal biasing")
#     print(f"   - Increase max iterations")


# if rrt_hard_success:
#     print("\n" + "-"*80)
#     print("STAGE 2: OPTIMIZATION (HARD)")
#     print("-"*80)

#     opt_hard_start = time.time()
#     optimized_hard, success_hard, info_hard = helpers.optimize_racing_path_sequential(
#         rrt_path=rrt_hard_path,
#         start_pose=start_hard,
#         hoops=hoops_hard,
#         obstacles=obstacles_hard,
#         N=30,
#         dt=0.1,
#         weights={'thrust': 0.03, 'angular': 0.3, 'smoothness': 8.0},
#         # Pass student-implemented functions (TODOs 17-28)
#         initialize_from_rrt_robust=rrt.initialize_from_rrt_robust,
#         dynamics_constraints_robust=boundary_constraints.dynamics_constraints_robust,
#         boundary_constraints_robust=boundary_constraints.boundary_constraints_robust,
#         collision_constraints_optimized=boundary_constraints.collision_constraints_optimized,
#         cost_function_tuned=cost_function.cost_function_tuned,
#         unpack_decision_vars=cost_function.unpack_decision_vars
#     )
#     opt_hard_time = time.time() - opt_hard_start

#     if success_hard:
#         print(f"\n🎉 HARD COURSE SUCCESS! 🎉")
#         print(f"   Time: {opt_hard_time:.2f}s")

#         has_collision_hard, _ = helpers.check_path_collision(optimized_hard, obstacles_hard)

#         print(f"   [✅] RRT succeeded")
#         print(f"   [{'✅' if success_hard else '❌'}] Optimization converged")
#         print(f"   [{'✅' if not has_collision_hard else '❌'}] Collision-free")
#         print(f"   [✅] Through all hoops")

#     else:
#         print(f"\n❌ Optimization failed for hard course")
#         print(f"   Partial credit for RRT success!")
#         optimized_hard = rrt_hard_path
# else:
#     print("\n⏭️ Skipping optimization (RRT failed)")

# if rrt_hard_success:
#     print("\n" + "-"*80)
#     print("STAGE 3: VISUALIZATION (HARD)")
#     print("-"*80)

#     fig_hard = helpers.drone_racing_path_with_obstacles(
#         hoops_hard, start_hard,
#         optimized_hard if success_hard else rrt_hard_path,
#         obstacles_hard,
#     )
#     fig_hard.show()
