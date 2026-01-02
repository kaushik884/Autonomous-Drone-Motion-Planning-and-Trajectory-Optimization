#!/usr/bin/env python3
"""
RViz2 Bridge for Drone Motion Planning Project

This module provides a simple interface to visualize your drone trajectories in RViz2.

Usage in your project:
    from rviz2_bridge import visualize_in_rviz2
    
    visualize_in_rviz2(
        hoops=hoops_final,
        start=start_final,
        rrt_path=rrt_final_path,
        optimized_path=optimized_final,
        obstacles=obstacles_easy
    )

Requirements:
    - ROS 2 Humble installed and sourced: source /opt/ros/humble/setup.bash
    - Run `rviz2` in a separate terminal

Author: Kaushik
"""

from __future__ import annotations

import numpy as np
import math
from typing import List, Optional, Any

# =============================================================================
# ROS 2 IMPORTS WITH PROPER ERROR HANDLING
# =============================================================================

ROS2_AVAILABLE = False

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from std_msgs.msg import Header, ColorRGBA
    from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Vector3
    from visualization_msgs.msg import Marker, MarkerArray
    from nav_msgs.msg import Path
    from tf2_ros import TransformBroadcaster
    from geometry_msgs.msg import TransformStamped
    ROS2_AVAILABLE = True
except ImportError as e:
    # Don't print warning at import time - only when actually used
    _ROS2_IMPORT_ERROR = str(e)

# Try to import gtsam
GTSAM_AVAILABLE = False
try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# HELPER FUNCTIONS (only defined if ROS2 available)
# =============================================================================

if ROS2_AVAILABLE:
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
        """Convert Euler angles (roll, pitch, yaw) to ROS quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q

    def rotation_matrix_to_quaternion(R: np.ndarray) -> Quaternion:
        """Convert 3x3 rotation matrix to ROS quaternion."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        q = Quaternion()
        q.w, q.x, q.y, q.z = w, x, y, z
        return q


    # =========================================================================
    # MAIN BRIDGE CLASS
    # =========================================================================

    class RViz2Bridge:
        """
        Bridge between your drone planning project and RViz2.
        
        This class handles all ROS 2 communication for visualization.
        """
        
        def __init__(self, node_name: str = 'drone_viz_bridge'):
            """Initialize the RViz2 bridge."""
            # Initialize ROS 2
            if not rclpy.ok():
                rclpy.init()
            
            self.node = rclpy.create_node(node_name)
            
            # QoS for latched topics
            latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            
            # Create publishers
            self.path_pub = self.node.create_publisher(Path, 'drone/path', latched_qos)
            self.rrt_path_pub = self.node.create_publisher(Path, 'drone/rrt_path', latched_qos)
            self.optimized_path_pub = self.node.create_publisher(Path, 'drone/optimized_path', latched_qos)
            self.pose_pub = self.node.create_publisher(PoseStamped, 'drone/pose', 10)
            self.hoops_pub = self.node.create_publisher(MarkerArray, 'racing/hoops', latched_qos)
            self.obstacles_pub = self.node.create_publisher(MarkerArray, 'racing/obstacles', latched_qos)
            self.rrt_tree_pub = self.node.create_publisher(Marker, 'drone/rrt_tree', latched_qos)
            self.waypoints_pub = self.node.create_publisher(MarkerArray, 'drone/waypoints', latched_qos)
            self.drone_mesh_pub = self.node.create_publisher(Marker, 'drone/mesh', 10)
            
            # TF broadcaster
            self.tf_broadcaster = TransformBroadcaster(self.node)
            
            # Animation state
            self._animation_path = []
            self._animation_idx = 0
            self._animation_timer = None
            
            self.node.get_logger().info('RViz2 Bridge initialized')
        
        def _create_header(self, frame_id: str = 'world') -> Header:
            """Create a message header."""
            header = Header()
            header.stamp = self.node.get_clock().now().to_msg()
            header.frame_id = frame_id
            return header
        
        def _pose_to_stamped(self, pose) -> PoseStamped:
            """Convert gtsam.Pose3 or state array to PoseStamped."""
            pose_stamped = PoseStamped()
            pose_stamped.header = self._create_header()
            
            if GTSAM_AVAILABLE and isinstance(pose, gtsam.Pose3):
                t = pose.translation()
                R = pose.rotation().matrix()
                pose_stamped.pose.position.x = float(t[0])
                pose_stamped.pose.position.y = float(t[1])
                pose_stamped.pose.position.z = float(t[2])
                pose_stamped.pose.orientation = rotation_matrix_to_quaternion(R)
            elif isinstance(pose, (list, np.ndarray)):
                pose_stamped.pose.position.x = float(pose[0])
                pose_stamped.pose.position.y = float(pose[1])
                pose_stamped.pose.position.z = float(pose[2])
                if len(pose) >= 6:
                    pose_stamped.pose.orientation = euler_to_quaternion(
                        float(pose[5]), float(pose[4]), float(pose[3])
                    )
                else:
                    pose_stamped.pose.orientation.w = 1.0
            else:
                raise TypeError(f"Unsupported pose type: {type(pose)}")
            
            return pose_stamped
        
        # =====================================================================
        # PATH PUBLISHING
        # =====================================================================
        
        def publish_path(self, poses: List, topic: str = 'main') -> None:
            """Publish a path to RViz2."""
            path_msg = Path()
            path_msg.header = self._create_header()
            
            for pose in poses:
                path_msg.poses.append(self._pose_to_stamped(pose))
            
            if topic == 'rrt':
                self.rrt_path_pub.publish(path_msg)
            elif topic == 'optimized':
                self.optimized_path_pub.publish(path_msg)
            else:
                self.path_pub.publish(path_msg)
            
            self.node.get_logger().info(f'Published {topic} path with {len(poses)} waypoints')
        
        def publish_path_comparison(self, rrt_path: List, optimized_path: List) -> None:
            """Publish both RRT and optimized paths for comparison."""
            self.publish_path(rrt_path, 'rrt')
            self.publish_path(optimized_path, 'optimized')
            self.publish_path(optimized_path, 'main')
        
        # =====================================================================
        # HOOPS
        # =====================================================================
        
        def publish_hoops(self, hoops: List) -> None:
            """Publish racing hoops."""
            marker_array = MarkerArray()
            
            for i, hoop in enumerate(hoops):
                marker = Marker()
                marker.header = self._create_header()
                marker.ns = 'hoops'
                marker.id = i
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                
                if GTSAM_AVAILABLE and isinstance(hoop, gtsam.Pose3):
                    t = hoop.translation()
                    R = hoop.rotation().matrix()
                else:
                    t = hoop[:3]
                    R = np.eye(3)
                
                radius = 1.0
                num_points = 32
                for j in range(num_points + 1):
                    angle = 2.0 * math.pi * j / num_points
                    local_pt = np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])
                    world_pt = R @ local_pt + np.array([t[0], t[1], t[2]])
                    marker.points.append(Point(x=float(world_pt[0]), y=float(world_pt[1]), z=float(world_pt[2])))
                
                marker.scale.x = 0.1
                marker.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)
                marker.pose.orientation.w = 1.0
                marker_array.markers.append(marker)
                
                # Label
                label = Marker()
                label.header = self._create_header()
                label.ns = 'hoop_labels'
                label.id = i + 100
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = float(t[0])
                label.pose.position.y = float(t[1])
                label.pose.position.z = float(t[2]) + 1.5
                label.pose.orientation.w = 1.0
                label.scale.z = 0.5
                label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                label.text = f'Hoop {i + 1}'
                marker_array.markers.append(label)
            
            self.hoops_pub.publish(marker_array)
            self.node.get_logger().info(f'Published {len(hoops)} hoops')
        
        # =====================================================================
        # OBSTACLES
        # =====================================================================
        
        def publish_obstacles(self, obstacles: List) -> None:
            """Publish obstacles (SphereObstacle, BoxObstacle)."""
            marker_array = MarkerArray()
            
            for i, obs in enumerate(obstacles):
                marker = Marker()
                marker.header = self._create_header()
                marker.ns = 'obstacles'
                marker.id = i
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                
                if hasattr(obs, 'radius'):
                    marker.type = Marker.SPHERE
                    center = obs.center
                    marker.pose.position.x = float(center[0])
                    marker.pose.position.y = float(center[1])
                    marker.pose.position.z = float(center[2])
                    d = 2.0 * obs.radius
                    marker.scale = Vector3(x=d, y=d, z=d)
                    marker.color = ColorRGBA(r=0.8, g=0.2, b=0.2, a=0.6)
                elif hasattr(obs, 'min_corner'):
                    marker.type = Marker.CUBE
                    min_c, max_c = obs.min_corner, obs.max_corner
                    marker.pose.position.x = float((min_c[0] + max_c[0]) / 2)
                    marker.pose.position.y = float((min_c[1] + max_c[1]) / 2)
                    marker.pose.position.z = float((min_c[2] + max_c[2]) / 2)
                    marker.scale.x = float(max_c[0] - min_c[0])
                    marker.scale.y = float(max_c[1] - min_c[1])
                    marker.scale.z = float(max_c[2] - min_c[2])
                    marker.color = ColorRGBA(r=0.6, g=0.1, b=0.1, a=0.6)
                
                marker_array.markers.append(marker)
            
            self.obstacles_pub.publish(marker_array)
            self.node.get_logger().info(f'Published {len(obstacles)} obstacles')
        
        # =====================================================================
        # START/GOAL
        # =====================================================================
        
        def publish_start_goal(self, start, goal) -> None:
            """Publish start and goal markers."""
            marker_array = MarkerArray()
            
            for idx, (pose, color, name) in enumerate([
                (start, ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9), 'START'),
                (goal, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9), 'GOAL')
            ]):
                if GTSAM_AVAILABLE and isinstance(pose, gtsam.Pose3):
                    pos = pose.translation()
                else:
                    pos = pose[:3] if hasattr(pose, '__len__') else pose
                
                marker = Marker()
                marker.header = self._create_header()
                marker.ns = 'start_goal'
                marker.id = idx * 2
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(pos[0])
                marker.pose.position.y = float(pos[1])
                marker.pose.position.z = float(pos[2])
                marker.pose.orientation.w = 1.0
                marker.scale = Vector3(x=0.5, y=0.5, z=0.5)
                marker.color = color
                marker_array.markers.append(marker)
                
                label = Marker()
                label.header = self._create_header()
                label.ns = 'start_goal'
                label.id = idx * 2 + 1
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = float(pos[0])
                label.pose.position.y = float(pos[1])
                label.pose.position.z = float(pos[2]) + 1.0
                label.pose.orientation.w = 1.0
                label.scale.z = 0.5
                label.color = color
                label.text = name
                marker_array.markers.append(label)
            
            self.waypoints_pub.publish(marker_array)
        
        # =====================================================================
        # RRT TREE
        # =====================================================================
        
        def publish_rrt_tree(self, rrt_nodes: List, parents: List[int]) -> None:
            """Publish RRT tree edges."""
            marker = Marker()
            marker.header = self._create_header()
            marker.ns = 'rrt_tree'
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color = ColorRGBA(r=0.5, g=0.5, b=1.0, a=0.4)
            marker.pose.orientation.w = 1.0
            
            for i, parent_idx in enumerate(parents):
                if parent_idx == -1:
                    continue
                
                child = rrt_nodes[i]
                parent = rrt_nodes[parent_idx]
                
                if GTSAM_AVAILABLE and isinstance(child, gtsam.Pose3):
                    child_pos = child.translation()
                    parent_pos = parent.translation()
                else:
                    child_pos = child[:3] if hasattr(child, '__len__') else child
                    parent_pos = parent[:3] if hasattr(parent, '__len__') else parent
                
                marker.points.append(Point(x=float(parent_pos[0]), y=float(parent_pos[1]), z=float(parent_pos[2])))
                marker.points.append(Point(x=float(child_pos[0]), y=float(child_pos[1]), z=float(child_pos[2])))
            
            self.rrt_tree_pub.publish(marker)
            self.node.get_logger().info(f'Published RRT tree with {len(rrt_nodes)} nodes')
        
        # =====================================================================
        # DRONE POSE & ANIMATION
        # =====================================================================
        
        def publish_drone_pose(self, pose) -> None:
            """Publish current drone pose with TF."""
            pose_stamped = self._pose_to_stamped(pose)
            self.pose_pub.publish(pose_stamped)
            
            t = TransformStamped()
            t.header = self._create_header()
            t.child_frame_id = 'drone'
            t.transform.translation.x = pose_stamped.pose.position.x
            t.transform.translation.y = pose_stamped.pose.position.y
            t.transform.translation.z = pose_stamped.pose.position.z
            t.transform.rotation = pose_stamped.pose.orientation
            self.tf_broadcaster.sendTransform(t)
            
            marker = Marker()
            marker.header = self._create_header()
            marker.ns = 'drone_body'
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = pose_stamped.pose
            marker.scale = Vector3(x=0.4, y=0.4, z=0.15)
            marker.color = ColorRGBA(r=0.2, g=0.6, b=0.8, a=0.9)
            self.drone_mesh_pub.publish(marker)
        
        def start_animation(self, path: List, rate_hz: float = 10.0) -> None:
            """Start animating the drone along a path."""
            self._animation_path = path
            self._animation_idx = 0
            period = 1.0 / rate_hz
            self._animation_timer = self.node.create_timer(period, self._animation_callback)
            self.node.get_logger().info(f'Started animation with {len(path)} poses')
        
        def _animation_callback(self) -> None:
            if not self._animation_path:
                return
            self.publish_drone_pose(self._animation_path[self._animation_idx])
            self._animation_idx = (self._animation_idx + 1) % len(self._animation_path)
        
        def stop_animation(self) -> None:
            if self._animation_timer:
                self._animation_timer.cancel()
                self._animation_timer = None
        
        # =====================================================================
        # HIGH-LEVEL API
        # =====================================================================
        
        def publish_racing_scenario(self, hoops: List, start, rrt_path: List,
                                     optimized_path: Optional[List] = None,
                                     obstacles: Optional[List] = None,
                                     animate: bool = True) -> None:
            """Publish a complete racing scenario."""
            self.publish_hoops(hoops)
            
            if obstacles:
                self.publish_obstacles(obstacles)
            
            if hoops:
                self.publish_start_goal(start, hoops[-1])
            
            self.publish_path(rrt_path, 'rrt')
            if optimized_path:
                self.publish_path(optimized_path, 'optimized')
                display_path = optimized_path
            else:
                display_path = rrt_path
            
            self.publish_path(display_path, 'main')
            
            if animate:
                self.start_animation(display_path)
        
        # =====================================================================
        # ROS 2 LIFECYCLE
        # =====================================================================
        
        def spin_once(self, timeout_sec: float = 0.1) -> None:
            """Process pending ROS 2 callbacks once."""
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)
        
        def spin(self) -> None:
            """Spin the node (blocking)."""
            try:
                rclpy.spin(self.node)
            except KeyboardInterrupt:
                pass
        
        def shutdown(self) -> None:
            """Clean shutdown."""
            self.stop_animation()
            self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


# =============================================================================
# CONVENIENCE FUNCTIONS (always defined)
# =============================================================================

def visualize_in_rviz2(hoops, start, rrt_path, optimized_path=None, obstacles=None,
                       animate=True, keep_alive=True):
    """
    One-liner to visualize your trajectory in RViz2.
    
    Usage in optimize.py:
        from rviz2_bridge import visualize_in_rviz2
        
        visualize_in_rviz2(
            hoops=hoops_final,
            start=start_final,
            rrt_path=rrt_final_path,
            optimized_path=optimized_final,
            obstacles=obstacles_easy
        )
    """
    if not ROS2_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: ROS 2 is not available!")
        print("="*60)
        print("\nTo fix this, run these commands in your terminal:")
        print("  source /opt/ros/humble/setup.bash")
        print("  python optimize.py")
        print("\nOr add this to your ~/.bashrc:")
        print("  echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc")
        print("  source ~/.bashrc")
        print("="*60 + "\n")
        return None
    
    bridge = RViz2Bridge()
    bridge.publish_racing_scenario(hoops, start, rrt_path, optimized_path, obstacles, animate)
    
    if keep_alive:
        print("\n" + "="*60)
        print("RViz2 visualization is running!")
        print("Open RViz2 in another terminal:")
        print("  rviz2")
        print("Or with config:")
        print("  rviz2 -d drone_racing.rviz")
        print("Press Ctrl+C to stop.")
        print("="*60 + "\n")
        bridge.spin()
    
    return bridge


def check_ros2() -> bool:
    """Check if ROS 2 is available and print helpful message if not."""
    if ROS2_AVAILABLE:
        print("✓ ROS 2 is available")
        return True
    else:
        print("✗ ROS 2 is NOT available")
        print("\nTo fix this:")
        print("  1. Make sure ROS 2 Humble is installed")
        print("  2. Source it: source /opt/ros/humble/setup.bash")
        print("  3. Run your script again")
        return False


if __name__ == '__main__':
    if not check_ros2():
        exit(1)
    
    print("\nTesting RViz2 Bridge...")
    print("Make sure RViz2 is running: rviz2")
    
    class MockPose3:
        def __init__(self, x, y, z):
            self._t = np.array([x, y, z])
        def translation(self):
            return self._t
        def rotation(self):
            return self
        def matrix(self):
            return np.eye(3)
    
    class MockObstacle:
        def __init__(self, center, radius, name):
            self.center = center
            self.radius = radius
            self.name = name
    
    hoops = [MockPose3(5, 5, 5), MockPose3(8, 10, 3), MockPose3(10, 12, 8), MockPose3(6, 7, 12)]
    start = MockPose3(1, 3, 8)
    path = [MockPose3(1+i*0.5, 3+i*0.4, 8-i*0.2) for i in range(20)]
    obstacles = [MockObstacle([5, 5, 7], 1.5, "Obs1")]
    
    visualize_in_rviz2(hoops, start, path, obstacles=obstacles)