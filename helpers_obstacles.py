from typing import List, Tuple
import math
import gtsam
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# ============================================================================
# OBSTACLE DEFINITIONS
# ============================================================================

'''The following functions and classes were taken from https://github.com/gtbook/gtbook/blob/master/drone.ipynb
This was done to remove the need to install the dependancy in colabs'''
def axes_figure(pose: gtsam.Pose3, scale: float = 1.0, labels: list = ["X", "Y", "Z"]):
    """Create plotly express figure with Pose3 coordinate frame."""
    t = np.reshape(pose.translation(),(3,1))
    M = np.hstack([t,t,t,pose.rotation().matrix() * scale + t])
    df = pd.DataFrame({"x": M[0], "y": M[1], "z": M[2]}, labels+labels)
    return px.line_3d(df, x="x", y="y", z="z", color=df.index,
                     color_discrete_sequence=["red", "green", "blue"])

def axes(*args, **kwargs):
    """Create 3 Scatter3d traces representing Pose3 coordinate frame."""
    return axes_figure(*args, **kwargs).data

'''Drone Kinematics and Dynamics explained in detail in : 
https://www.roboticsbook.org/S72_drone_actions.html'''
from dataclasses import dataclass, field
@dataclass
class DroneKinematics:
    rn: gtsam.Point3
    vn: gtsam.Point3
    nRb: gtsam.Rot3
    wb: gtsam.Point3

    def pose(self):
        """Return the current pose of the drone."""
        return gtsam.Pose3(self.nRb, self.rn)

    def integrate(self, dt):
        """Integrate the drone position/attitude forward in time by dt seconds."""
        self.rn += self.vn * dt  # integrate position
        # calculate incremental rotation matrix using the exponential map
        dR = gtsam.Rot3.Expmap(self.wb * dt)
        self.nRb = self.nRb * dR  # integrate attitude

@dataclass
class Drone(DroneKinematics):
    g: float = 9.81
    mass: float = 1.0
    k_d: float = 0.0425                        # drag coefficient
    gn = np.array([0, 0, -g])    # gravity vector in navigation frame.
    # gn: np.ndarray = field(default_factory=lambda:np.array([0,0,-9.81]))    # gravity vector in navigation frame.
    I_xy: float = 4 * 0.15 * mass * 0.1**2     # All 4 motors contribute to I_xy
    I_z: float = 4 * 0.15 * mass * 2 * 0.1**2  # I_z leverages longer arm
    I = np.diag([I_xy, I_xy, I_z])             # Inertia matrix

    # define integrate method that updates dynamics and then calls super's integrate method:
    def integrate(self, f: float, tau:gtsam.Point3, dt=1.0):
        """Integrate equations of motion given dynamic inputs f and tau."""
        # Calculate net force in navigation frame, including gravity and drag:
        Fb = gtsam.Point3(0, 0, f)
        drag_force = -self.k_d * self.vn * np.linalg.norm(self.vn)
        net_force = self.nRb.rotate(Fb) + self.gn + drag_force

        # Integrate velocity in navigation frame:
        self.vn += net_force * dt / self.mass

        # rotational drag, assume 10x smaller linearly proportional to angular velocity:
        net_tau = tau - 0.1 * self.k_d * self.wb

        # Integrate angular velocity in body frame:
        self.wb[0] += net_tau[0] * dt / self.I_xy
        self.wb[1] += net_tau[1] * dt / self.I_xy
        self.wb[2] += net_tau[2] * dt / self.I_z

        # Call super's integrate method for kinematics
        super().integrate(dt)


class Obstacle:
    """Base class for obstacles"""
    def __init__(self, name: str):
        self.name = name
    
    def is_in_collision(self, point: gtsam.Point3) -> bool:
        raise NotImplementedError


class SphereObstacle(Obstacle):
    """Spherical obstacle"""
    def __init__(self, center: gtsam.Point3, radius: float, name: str = "Sphere"):
        super().__init__(name)
        self.center = center
        self.radius = radius
    
    def is_in_collision(self, point: gtsam.Point3) -> bool:
        """Check if point is inside sphere"""
        distance = np.linalg.norm(point - self.center)
        return distance < self.radius
    
    def get_mesh_data(self, resolution: int = 20):
        """Generate sphere mesh for visualization"""
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = self.center[0] + self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.center[1] + self.radius * np.outer(np.sin(u), np.sin(v))
        z = self.center[2] + self.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z


class BoxObstacle(Obstacle):
    """Rectangular box obstacle"""
    def __init__(self, min_corner: gtsam.Point3, max_corner: gtsam.Point3, name: str = "Box"):
        super().__init__(name)
        self.min_corner = min_corner
        self.max_corner = max_corner
    
    def is_in_collision(self, point: gtsam.Point3) -> bool:
        """Check if point is inside box"""
        return (self.min_corner[0] <= point[0] <= self.max_corner[0] and
                self.min_corner[1] <= point[1] <= self.max_corner[1] and
                self.min_corner[2] <= point[2] <= self.max_corner[2])
    
    def get_mesh_data(self):
        """Generate box edges for visualization"""
        min_x, min_y, min_z = self.min_corner[0], self.min_corner[1], self.min_corner[2]
        max_x, max_y, max_z = self.max_corner[0], self.max_corner[1], self.max_corner[2]
        
        # 12 edges of the box
        edges = [
            # Bottom face
            ([min_x, max_x], [min_y, min_y], [min_z, min_z]),
            ([min_x, max_x], [max_y, max_y], [min_z, min_z]),
            ([min_x, min_x], [min_y, max_y], [min_z, min_z]),
            ([max_x, max_x], [min_y, max_y], [min_z, min_z]),
            # Top face
            ([min_x, max_x], [min_y, min_y], [max_z, max_z]),
            ([min_x, max_x], [max_y, max_y], [max_z, max_z]),
            ([min_x, min_x], [min_y, max_y], [max_z, max_z]),
            ([max_x, max_x], [min_y, max_y], [max_z, max_z]),
            # Vertical edges
            ([min_x, min_x], [min_y, min_y], [min_z, max_z]),
            ([max_x, max_x], [min_y, min_y], [min_z, max_z]),
            ([min_x, min_x], [max_y, max_y], [min_z, max_z]),
            ([max_x, max_x], [max_y, max_y], [min_z, max_z]),
        ]
        return edges


# ============================================================================
# COLLISION CHECKING FUNCTIONS
# ============================================================================

def check_point_collision(point: gtsam.Point3, obstacles: List[Obstacle]) -> bool:
    """
    Check if a single point collides with any obstacle.
    
    Arguments:
     - point: gtsam.Point3, position to check
     - obstacles: List[Obstacle], list of obstacles
    
    Returns:
     - collision: bool, True if point is in collision
    """
    for obstacle in obstacles:
        if obstacle.is_in_collision(point):
            return True
    return False


def check_segment_collision(point1: gtsam.Point3, point2: gtsam.Point3, 
                           obstacles: List[Obstacle], num_checks: int = 10) -> bool:
    """
    Check if line segment between two points collides with any obstacle.
    
    Uses linear interpolation to check multiple points along the segment.
    
    Arguments:
     - point1: gtsam.Point3, start of segment
     - point2: gtsam.Point3, end of segment
     - obstacles: List[Obstacle], list of obstacles
     - num_checks: int, number of points to check along segment
    
    Returns:
     - collision: bool, True if segment intersects any obstacle
    """
    # Check endpoints first
    if check_point_collision(point1, obstacles) or check_point_collision(point2, obstacles):
        return True
    
    # Check intermediate points
    for i in range(1, num_checks):
        alpha = i / num_checks
        # Linear interpolation: p = (1-α)p1 + αp2
        intermediate = point1 + alpha * (point2 - point1)
        if check_point_collision(intermediate, obstacles):
            return True
    
    return False


def check_path_collision(path: List, obstacles: List[Obstacle]) -> Tuple[bool, List[int]]:
    """
    Check if entire path collides with any obstacle.
    
    Arguments:
     - path: List[gtsam.Point3] or List[gtsam.Pose3], path to check
     - obstacles: List[Obstacle], list of obstacles
    
    Returns:
     - has_collision: bool, True if any segment collides
     - collision_segments: List[int], indices of colliding segments
    """
    collision_segments = []
    
    for i in range(len(path) - 1):
        # Extract positions
        if isinstance(path[0], gtsam.Pose3):
            p1 = path[i].translation()
            p2 = path[i+1].translation()
        else:
            p1 = path[i]
            p2 = path[i+1]
        
        # Check segment
        if check_segment_collision(p1, p2, obstacles):
            collision_segments.append(i)
    
    return len(collision_segments) > 0, collision_segments


# ============================================================================
# OBSTACLE CONFIGURATIONS FOR RACING
# ============================================================================

def get_obstacles_easy() -> List[Obstacle]:
    """
    Easy obstacle course - 2 spherical obstacles between hoops
    """
    obstacles = [
        # Obstacle between start and hoop 1
        SphereObstacle(
            center=gtsam.Point3(4.5, 4.0, 7.0),
            radius=1.5,
            name="Obstacle 1"
        ),
        # Obstacle between hoop 2 and hoop 3
        SphereObstacle(
            center=gtsam.Point3(8.0, 11.5, 6.5),
            radius=1.2,
            name="Obstacle 2"
        ),
    ]
    return obstacles


# def get_obstacles_medium() -> List[Obstacle]:
#     """
#     Medium difficulty - mix of spheres and boxes
#     """
#     obstacles = [
#         # Sphere between start and hoop 1
#         SphereObstacle(
#             center=gtsam.Point3(4.5, 4.0, 7.0),
#             radius=1.5,
#             name="Sphere 1"
#         ),
#         # Box blocking direct path to hoop 2
#         # BoxObstacle(
#         #     min_corner=gtsam.Point3(7.0, 8.0, 1.0),
#         #     max_corner=gtsam.Point3(9.0, 10.0, 4.0),
#         #     name="Box 1"
#         # ),
#         # Sphere near hoop 3
#         SphereObstacle(
#             center=gtsam.Point3(8.0, 11.5, 10.0),
#             radius=1.0,
#             name="Sphere 2"
#         ),
#         # Small obstacle near hoop 4
#         SphereObstacle(
#             center=gtsam.Point3(6.0, 6.0, 12.0),
#             radius=0.8,
#             name="Sphere 3"
#         ),
#     ]
#     return obstacles


def get_obstacles_hard() -> List[Obstacle]:
    """
    Hard obstacle course - many obstacles requiring careful navigation
    """
    obstacles = [
        # Multiple spheres creating a maze-like structure
        SphereObstacle(gtsam.Point3(3.0, 3.0, 7.0), 1.2, "S1"),
        SphereObstacle(gtsam.Point3(6.0, 4.0, 6.0), 1.0, "S2"),
        SphereObstacle(gtsam.Point3(8.0, 8.0, 3.0), 1.5, "S3"),
        SphereObstacle(gtsam.Point3(8.0, 11.0, 8.0), 1.3, "S4"),
        SphereObstacle(gtsam.Point3(6.5, 8.0, 11.0), 1.0, "S5"),
        SphereObstacle(gtsam.Point3(5.5, 6.5, 11.5), 0.8, "S6"),
        
        # Boxes creating walls
        BoxObstacle(gtsam.Point3(7.5, 7.0, 0.0), gtsam.Point3(9.0, 9.0, 1.5), "B1"),
        BoxObstacle(gtsam.Point3(4.0, 4.5, 9.0), gtsam.Point3(6.0, 5.5, 12.0), "B2"),
    ]
    return obstacles


# ============================================================================
# VISUALIZATION WITH OBSTACLES
# ============================================================================

def visualize_obstacles(obstacles: List[Obstacle], show_labels: bool = True):
    """
    Visualize obstacles in 3D space
    """
    fig = go.Figure()
    
    for obs in obstacles:
        if isinstance(obs, SphereObstacle):
            x, y, z = obs.get_mesh_data()
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.5,
                colorscale='Reds',
                showscale=False,
                name=obs.name,
                hovertext=obs.name
            ))
        elif isinstance(obs, BoxObstacle):
            edges = obs.get_mesh_data()
            for edge_x, edge_y, edge_z in edges:
                fig.add_trace(go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode='lines',
                    line=dict(color='red', width=4),
                    showlegend=False,
                    hovertext=obs.name
                ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 15]),
            yaxis=dict(range=[0, 15]),
            zaxis=dict(range=[0, 15]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        title="Obstacle Configuration",
        width=1000,
        height=700
    )
    
    fig.show()


def drone_racing_path_with_obstacles(hoops, start, path, obstacles: List[Obstacle] = None, 
                                     show_grid_lines=True, duration=0.3):
    """
    Enhanced drone racing visualization with obstacles.
    
    Arguments:
     - hoops: List of hoop poses
     - start: gtsam.Pose3, start position
     - path: List[gtsam.Pose3], drone path
     - obstacles: List[Obstacle], obstacles to visualize (optional)
     - show_grid_lines: bool, show grid
     - duration: float, animation duration
    """
    num_steps = len(path)
    labels = [str(i) for i in range(num_steps)]
    first_data = None
    frames = []
    
    for frame_idx in range(num_steps):
        xs_gt = []
        ys_gt = []
        zs_gt = []
        for pose_idx in range(frame_idx+1):
            node = path[pose_idx]
            x, y, z = node.translation()
            xs_gt.append(x)
            ys_gt.append(y)
            zs_gt.append(z)
        
        data = list(axes(path[frame_idx], scale=1.0, labels=['F', 'L', 'U']))
        data.append(go.Scatter3d(
            x=xs_gt, y=ys_gt, z=zs_gt,
            mode='lines+markers',
            marker=dict(size=3),
            showlegend=False,
        ))
        frames.append(go.Frame(name=labels[frame_idx], data=data))
        if first_data is None:
            first_data = data
    
    play_button = generate_play_button(num_steps, duration)
    layout = generate_layout(play_button, labels, duration, show_grid_lines)
    fig = go.Figure(data=first_data, layout=layout, frames=frames)
    
    # Set ranges
    fig.update_layout(    
        scene = dict(
            xaxis=dict(range=[0, 15], autorange=False),
            yaxis=dict(range=[0, 15], autorange=False),
            zaxis=dict(range=[0, 15], autorange=False),
        ))
    
    fig.layout.scene.aspectratio = {'x':1, 'y':1, 'z':1}
    
    # Add start point
    fig.add_scatter3d(x=[start.x()], y=[start.y()], z=[start.z()], 
                  mode="markers", marker=dict(color="yellow", size=10), name="Start")
    
    # Add hoops
    for count, hoop in enumerate(hoops):  
        xcrd, ycrd, zcrd = draw_hoop(hoop)
        fig.add_scatter3d(x=xcrd, y=ycrd, z=zcrd, marker=dict(size=0.1), 
                         line=dict(color='black', width=5), name=f"Hoop {count + 1}")
    
    # Add obstacles if provided
    if obstacles is not None:
        for obs in obstacles:
            if isinstance(obs, SphereObstacle):
                x, y, z = obs.get_mesh_data(resolution=15)
                fig.add_trace(go.Surface(
                    x=x, y=y, z=z,
                    opacity=0.3,
                    colorscale='Reds',
                    showscale=False,
                    name=obs.name,
                    hovertext=f"{obs.name} (r={obs.radius:.1f}m)"
                ))
            elif isinstance(obs, BoxObstacle):
                edges = obs.get_mesh_data()
                for edge_x, edge_y, edge_z in edges:
                    fig.add_trace(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color='darkred', width=3),
                        showlegend=False,
                        hovertext=obs.name
                    ))
    
    fig.show()


def visualize_path_with_obstacles(path, start, target, obstacles: List[Obstacle] = None,
                                  show_grid_lines=True, duration=0.3):
    """
    Visualize simple path with obstacles (for testing individual segments)
    """
    num_steps = len(path)
    
    xs_gt = []
    ys_gt = []
    zs_gt = []
    for pose_idx in range(num_steps):
        node = path[pose_idx]
        if isinstance(node, gtsam.Pose3):
            x, y, z = node.translation()
        else:
            x, y, z = node
        xs_gt.append(x)
        ys_gt.append(y)
        zs_gt.append(z)
    
    d = {'X Axis': xs_gt, 'Y Axis': ys_gt, 'Z Axis': zs_gt}
    df = pd.DataFrame(data=d)
    fig = px.line_3d(df, x='X Axis', y='Y Axis', z='Z Axis', markers=True, width=1200, height=800)
    
    # Add start and target
    if isinstance(target, gtsam.Pose3):
        fig.add_scatter3d(x=[target.x()], y=[target.y()], z=[target.z()], 
                    mode="markers", marker=dict(color="green", size=10), name="Target")
        fig.add_scatter3d(x=[start.x()], y=[start.y()], z=[start.z()], 
                    mode="markers", marker=dict(color="red", size=10), name="Start")
    else:
        fig.add_scatter3d(x=[target[0]], y=[target[1]], z=[target[2]], 
                    mode="markers", marker=dict(color="green", size=10), name="Target")
        fig.add_scatter3d(x=[start[0]], y=[start[1]], z=[start[2]], 
                    mode="markers", marker=dict(color="red", size=10), name="Start")
    
    # Add obstacles
    if obstacles is not None:
        for obs in obstacles:
            if isinstance(obs, SphereObstacle):
                x, y, z = obs.get_mesh_data(resolution=15)
                fig.add_trace(go.Surface(
                    x=x, y=y, z=z,
                    opacity=0.4,
                    colorscale='Reds',
                    showscale=False,
                    name=obs.name
                ))
            elif isinstance(obs, BoxObstacle):
                edges = obs.get_mesh_data()
                for edge_x, edge_y, edge_z in edges:
                    fig.add_trace(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color='darkred', width=3),
                        showlegend=False
                    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10]),
            zaxis=dict(range=[0, 10]),
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    
    fig.show()


# ============================================================================
# COMPARISON VISUALIZATION
# ============================================================================

def visualize_rrt_vs_optimized_with_obstacles(hoops, start, rrt_path, optimized_path,
                                             obstacles: List[Obstacle] = None):
    """
    Side-by-side comparison of RRT and optimized paths with obstacles shown.
    """
    fig = go.Figure()
    
    # Start point
    fig.add_trace(go.Scatter3d(
        x=[start.x()], y=[start.y()], z=[start.z()],
        mode='markers',
        marker=dict(size=10, color='yellow'),
        name='Start'
    ))
    
    # Hoops
    for i, hoop in enumerate(hoops):
        xcrd, ycrd, zcrd = draw_hoop(hoop)
        fig.add_trace(go.Scatter3d(
            x=xcrd, y=ycrd, z=zcrd,
            mode='lines',
            line=dict(color='black', width=5),
            name=f'Hoop {i+1}',
            showlegend=(i==0)
        ))
    
    # RRT path
    if rrt_path and len(rrt_path) > 0:
        rrt_x = [p.translation()[0] for p in rrt_path]
        rrt_y = [p.translation()[1] for p in rrt_path]
        rrt_z = [p.translation()[2] for p in rrt_path]
        fig.add_trace(go.Scatter3d(
            x=rrt_x, y=rrt_y, z=rrt_z,
            mode='lines',
            line=dict(color='cyan', width=4),
            name='RRT Path'
        ))
    
    # Optimized path
    if optimized_path and len(optimized_path) > 0:
        opt_x = [p.translation()[0] for p in optimized_path]
        opt_y = [p.translation()[1] for p in optimized_path]
        opt_z = [p.translation()[2] for p in optimized_path]
        fig.add_trace(go.Scatter3d(
            x=opt_x, y=opt_y, z=opt_z,
            mode='lines',
            line=dict(color='magenta', width=4),
            name='Optimized Path'
        ))
    
    # Add obstacles
    if obstacles is not None:
        for obs in obstacles:
            if isinstance(obs, SphereObstacle):
                x, y, z = obs.get_mesh_data(resolution=20)
                fig.add_trace(go.Surface(
                    x=x, y=y, z=z,
                    opacity=0.3,
                    colorscale='Reds',
                    showscale=False,
                    name=obs.name
                ))
            elif isinstance(obs, BoxObstacle):
                edges = obs.get_mesh_data()
                for edge_x, edge_y, edge_z in edges:
                    fig.add_trace(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color='darkred', width=4),
                        showlegend=False
                    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 15]),
            yaxis=dict(range=[0, 15]),
            zaxis=dict(range=[0, 15]),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=-1.25, y=2, z=0.5))
        ),
        title="RRT vs Optimized: Obstacle Avoidance Comparison",
        width=1400,
        height=900
    )
    
    fig.show()


# ============================================================================
# KEEP EXISTING HELPER FUNCTIONS
# ============================================================================

def visualize_tree(rrt, parents, start, target):
    xline, yline, zline, group = [], [], [], []
    for i, node in enumerate(rrt):
        if(i==0):
            continue
        if(type(rrt[0]) == np.ndarray):
            x, y, z = rrt[parents[i]]
        else:
            x, y, z = rrt[parents[i]].translation()
        xline.append(x)
        yline.append(y)
        zline.append(z)
        
        if(type(rrt[0]) == np.ndarray):
            x, y, z = node
        else: 
            x, y, z = node.translation()
        xline.append(x)
        yline.append(y)
        zline.append(z)
        group.append(i)
        group.append(i)
    
    fig = px.line_3d(x=xline, y=yline, z=zline, line_group=group, markers='true', width=1200, height=800)
    
    if(type(rrt[0]) == np.ndarray):
        fig.add_scatter3d(x=[start[0]], y=[start[1]], z=[start[2]], 
                      mode="markers", marker=dict(color="red"), name="Start")
        fig.add_scatter3d(x=[target[0]], y=[target[1]], z=[target[2]], 
                      mode="markers", marker=dict(color="green"), name="Target")
    else: 
        fig.add_scatter3d(x=[start.x()], y=[start.y()], z=[start.z()], 
                      mode="markers", marker=dict(color="red"), name="Start")
        fig.add_scatter3d(x=[target.x()], y=[target.y()], z=[target.z()], 
                      mode="markers", marker=dict(color="green"), name="Target")
    fig.show()


def visualize_path(path, start, target, show_grid_lines=True, duration=0.3):
    num_steps = len(path)
    xs_gt = []
    ys_gt = []
    zs_gt = []
    for pose_idx in range(num_steps):
        node = path[pose_idx]
        if(type(path[0]) == np.ndarray):
            x, y, z = node
        else:
            x, y, z = node.translation()
        xs_gt.append(x)
        ys_gt.append(y)
        zs_gt.append(z)
    
    d = {'X Axis': xs_gt, 'Y Axis': ys_gt, 'Z Axis': zs_gt}
    df = pd.DataFrame(data=d)
    fig1 = px.line_3d(df, x='X Axis', y='Y Axis', z='Z Axis', markers='true', width=1200, height=800)
    
    if(type(path[0]) == np.ndarray):
        fig1.add_scatter3d(x=[target[0]], y=[target[1]], z=[target[2]], 
                      mode="markers", marker=dict(color="green"), name="Target")
        fig1.add_scatter3d(x=[start[0]], y=[start[1]], z=[start[2]], 
                      mode="markers", marker=dict(color="red"), name="Start")
    else:
        fig1.add_scatter3d(x=[target.x()], y=[target.y()], z=[target.z()], 
                      mode="markers", marker=dict(color="green"), name="Target")
        fig1.add_scatter3d(x=[start.x()], y=[start.y()], z=[start.z()], 
                      mode="markers", marker=dict(color="red"), name="Start")
    
    fig1.show()


def distance_between_poses(pose1: gtsam.Pose3, pose2: gtsam.Pose3) -> float:
    weight = np.array([1, 1, 1, 1, 1, 1], float)
    return np.linalg.norm(weight * gtsam.Pose3.logmap(pose1, pose2))


def draw_hoop(hoop_pose: gtsam.Pose3):
    dist = 0.5
    orbit_points = []
    for i in range(0, 361):
        point = gtsam.Point3(
            (round(np.cos(math.radians(i)), 5)) * dist,
            (round(np.sin(math.radians(i)), 5)) * dist,
            0
        )
        orbit_points.append(hoop_pose.transformFrom(point))
    orbit_points = np.array(orbit_points)
    return orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2]


def generate_play_button(num_steps, duration):
    play_button = []
    if num_steps >= 2:
        play_button = [dict(
            buttons=[dict(
                args=[None, dict(
                    frame=dict(duration=1000 * duration),
                    fromcurrent=True
                )],
                label="Play",
                method="animate"
            )],
            direction="left",
            pad=dict(r=10, t=30),
            showactive=False,
            type="buttons",
            x=0.05,
            xanchor="right",
            y=0,
            yanchor="top",
            font=dict(
                family="Courier New, monospace",
                color='rgb(230, 230, 230)'
            )
        )]
    return play_button


def generate_layout(play_button, labels, duration, show_grid_lines):
    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5
    layout = go.Layout(
        margin=dict(r=50, l=50, b=30, t=30),
        width=1000,
        height=700,
        paper_bgcolor='rgba(50, 50, 60, 255)',
        plot_bgcolor='rgba(50, 50, 60, 255)',
        font=dict(
            family="Courier New, monospace",
            color='rgba(230, 230, 230, 255)' if show_grid_lines else 'rgba(255, 255, 255, 0)'
        ),
        hovermode=False,
        sliders=[dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(
                    family="Courier New, monospace",
                    color='rgb(230, 230, 230)'
                ),
                prefix="Step ",
                visible=True,
                xanchor="right"
            ),
            pad=dict(b=10, t=0),
            len=0.95,
            x=0.05,
            y=0,
            font=dict(
                family="Courier New, monospace",
                color='rgb(230, 230, 230)'
            ),
            steps=[dict(
                    args=[[local_label], dict(
                        frame=dict(duration=1000 * duration),
                        mode="immediate"
                    )],
                    label=local_label,
                    method="animate") for local_label in labels]
        )],
        updatemenus=play_button,
        scene=dict(camera=dict(eye=dict(x=x_eye, y=y_eye, z=z_eye))),
    )
    return layout


def animate_drone_path(path, start, target, show_grid_lines=True, duration=0.3):
    num_steps = len(path)
    labels = [str(i) for i in range(num_steps)]
    first_data = None
    frames = []
    for frame_idx in range(num_steps):
        xs_gt = []
        ys_gt = []
        zs_gt = []
        for pose_idx in range(frame_idx+1):
            node = path[pose_idx]
            x, y, z = node.translation()
            xs_gt.append(x)
            ys_gt.append(y)
            zs_gt.append(z)
        data = list(axes(path[frame_idx], scale=1.0, labels=['F', 'L', 'U']))
        data.append(go.Scatter3d(
                x=xs_gt, y=ys_gt, z=zs_gt,
                mode='lines+markers',
                marker=dict(size=7),
                showlegend=False,
        ))
        frames.append(go.Frame(name=labels[frame_idx], data=data))
        if first_data is None:
            first_data = data
    
    play_button = generate_play_button(num_steps, duration)
    layout = generate_layout(play_button, labels, duration, show_grid_lines)
    fig = go.Figure(data=first_data, layout=layout, frames=frames)
    
    fig.update_layout(    
        scene = dict(
            xaxis=dict(range=[0, 10], autorange=False),
            yaxis=dict(range=[0, 10], autorange=False),
            zaxis=dict(range=[0, 10], autorange=False),
        ))
    
    fig.layout.scene.aspectratio = {'x':1, 'y':1, 'z':1}
    
    fig.add_scatter3d(x=[start.x()], y=[start.y()], z=[start.z()], 
                  mode="markers", marker=dict(color="yellow"), name="Start")
    fig.add_scatter3d(x=[target.x()], y=[target.y()], z=[target.z()], 
                  mode="markers", marker=dict(color="purple"), name="Target")
    
    xcrd, ycrd, zcrd = draw_hoop(target)
    fig.add_scatter3d(x=xcrd, y=ycrd, z=zcrd, marker=dict(size=0.1), 
                     line=dict(color='black', width=5), name="Hoop")
    
    fig.show()


def get_new_attitude(current: gtsam.Pose3, direction: gtsam.Point3):
    f = np.cross(direction, current.rotation().column(3))
    l = np.cross(direction, f)
    return gtsam.Rot3(f, l, direction)


def drone_racing_path(hoops, start, path, show_grid_lines=True, duration=0.3):
    """
    DEPRECATED: Use drone_racing_path_with_obstacles instead.
    Kept for backward compatibility.
    """
    drone_racing_path_with_obstacles(hoops, start, path, obstacles=[], 
                                    show_grid_lines=show_grid_lines, duration=duration)


def pass_through_the_hoop(target: gtsam.Pose3, path: list):
    path.append(target)
    path.append(gtsam.Pose3(target.rotation(), target.transformFrom([0, 0, 1])))
    path.append(gtsam.Pose3(target.rotation(), target.transformFrom([0, 0, 2])))


def get_hoops():
    return [gtsam.Pose3(r=gtsam.Rot3.Ypr(0, math.radians(45), 0), t=gtsam.Point3(8, 5, 6)),
            gtsam.Pose3(r=gtsam.Rot3.Ypr(0, math.radians(45), math.radians(45)), t=gtsam.Point3(8, 13, 2)),
            gtsam.Pose3(r=gtsam.Rot3.Ypr(0, math.radians(30), math.radians(30)), t=gtsam.Point3(8, 10, 11)),
            gtsam.Pose3(r=gtsam.Rot3.Ypr(0, math.radians(45), math.radians(45)), t=gtsam.Point3(5, 5, 13))]


def get_targets() -> list:
    return [gtsam.Pose3(target.rotation(), target.transformFrom([0, 0, -1])) for target in get_hoops()]


# ============================================================================
# OPTIMIZATION UTILITIES (Black-box helpers - students use but don't implement)
# ============================================================================

def rotation_matrix_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Compute 3x3 rotation matrix from Euler angles (ZYX aerospace convention).
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Arguments:
        yaw: rotation about z-axis (radians)
        pitch: rotation about y-axis (radians)
        roll: rotation about x-axis (radians)

    Returns:
        R: 3x3 rotation matrix
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ])
    return R


def euler_from_rotation_matrix_safe(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract ZYX Euler angles from rotation matrix with gimbal lock handling.

    Arguments:
        R: 3x3 rotation matrix

    Returns:
        (yaw, pitch, roll) in radians
    """
    # Clamp to handle numerical errors
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))

    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: set roll=0, solve for yaw
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    return yaw, pitch, roll


def find_hoop_indices_robust(positions: np.ndarray, hoop_positions: List, N: int) -> List[int]:
    """
    Find indices in downsampled trajectory closest to each hoop.
    Ensures indices are ordered and well-separated.

    Arguments:
        positions: (N+1) x 3 array of positions
        hoop_positions: list of hoop positions
        N: number of time steps

    Returns:
        hoop_indices: list of knot indices for each hoop
    """
    hoop_indices = []
    min_spacing = max(1, N // (len(hoop_positions) + 2))

    for h_idx, hoop_pos in enumerate(hoop_positions):
        # Find closest knot to this hoop
        distances = [np.linalg.norm(positions[k] - hoop_pos) for k in range(N + 1)]
        best_k = int(np.argmin(distances))

        # Enforce ordering and spacing
        if hoop_indices:
            best_k = max(best_k, hoop_indices[-1] + min_spacing)

        # Clamp to valid range (leave room for remaining hoops)
        best_k = min(best_k, N - (len(hoop_positions) - h_idx) * min_spacing)

        hoop_indices.append(best_k)

    return hoop_indices


def convert_states_to_poses(states: np.ndarray) -> List[gtsam.Pose3]:
    """
    Convert state array to list of gtsam.Pose3.

    Arguments:
        states: (N+1) x 6 array [px, py, pz, yaw, pitch, roll]

    Returns:
        path: list of Pose3 objects
    """
    path = []
    for k in range(len(states)):
        px, py, pz, yaw, pitch, roll = states[k, :]
        R = rotation_matrix_from_euler(yaw, pitch, roll)
        rot = gtsam.Rot3(R)
        pos = gtsam.Point3(px, py, pz)
        path.append(gtsam.Pose3(rot, pos))
    return path

# ============================================================================
# TRAJECTORY OPTIMIZATION (PROVIDED - Black-box for students)
# ============================================================================

def optimize_trajectory(
    rrt_path: List[gtsam.Pose3],
    start_pose: gtsam.Pose3,
    goal_position: np.ndarray,
    hoops: List[gtsam.Pose3],
    obstacles: List,
    N: int = 30,
    dt: float = 0.1,
    weights: dict = None,
    # Student-implemented functions passed as parameters
    initialize_from_rrt_robust=None,
    dynamics_constraints_robust=None,
    boundary_constraints_robust=None,
    collision_constraints_optimized=None,
    cost_function_tuned=None,
    convert_states_to_poses=convert_states_to_poses,
    unpack_decision_vars=None
) -> Tuple[List[gtsam.Pose3], bool, dict]:
    """
    Optimize trajectory using direct transcription (PROVIDED FUNCTION).

    This function is provided as working code because it involves complex scipy.optimize
    API usage and orchestration. Students implement the cost functions and constraints
    (TODOs 17-28) which are passed to this function as parameters.

    Arguments:
        rrt_path: Initial path from RRT
        start_pose: Starting pose
        goal_position: Goal position (3D point)
        hoops: List of hoop poses to pass through
        obstacles: List of obstacles to avoid
        N: Number of time steps
        dt: Time step duration
        weights: Cost function weights

        # Student-implemented functions (required):
        initialize_from_rrt_robust: Function to initialize decision variables from RRT
        dynamics_constraints_robust: Function enforcing dynamics constraints
        boundary_constraints_robust: Function enforcing boundary constraints
        collision_constraints_optimized: Function enforcing collision constraints
        cost_function_tuned: Combined cost function
        convert_states_to_poses: Function to convert states to poses
        unpack_decision_vars: Function to unpack decision variables

    Returns:
        optimized_path: Optimized trajectory
        success: Whether optimization succeeded
        info: Diagnostic information
    """
    from scipy.optimize import minimize

    # Validate that required functions are provided
    required_funcs = {
        'initialize_from_rrt_robust': initialize_from_rrt_robust,
        'dynamics_constraints_robust': dynamics_constraints_robust,
        'boundary_constraints_robust': boundary_constraints_robust,
        'cost_function_tuned': cost_function_tuned,
        'convert_states_to_poses': convert_states_to_poses,
        'unpack_decision_vars': unpack_decision_vars
    }

    missing = [name for name, func in required_funcs.items() if func is None]
    if missing:
        raise ValueError(f"Missing required functions: {', '.join(missing)}\n"
                        f"These should be implemented in TODOs 17-28 and passed to optimize_trajectory()")
    
    if weights is None:
        weights = {'thrust': 0.1, 'angular': 1.0, 'smoothness': 5.0}
    
    print(f"\n{'='*60}")
    print(f"TRAJECTORY OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Knot points: {N}, Time step: {dt}s, Duration: {N*dt:.1f}s")
    print(f"Decision variables: {10*N + 6}")
    print(f"{'='*60}\n")
    
    # Initialize
    print("Initializing from RRT...")
    z_init = initialize_from_rrt_robust(rrt_path, N, dt, start_pose)
    
    # Set bounds
    deg_to_rad = np.pi / 180
    bounds = []
    for k in range(N + 1):
        bounds.extend([
            (0, 15), (0, 15), (0, 15),  # position
            (-60*deg_to_rad, 60*deg_to_rad),  # yaw
            (-60*deg_to_rad, 60*deg_to_rad),  # pitch
            (-60*deg_to_rad, 60*deg_to_rad),  # roll
        ])
    for k in range(N):
        bounds.extend([
            (-10*deg_to_rad, 10*deg_to_rad),  # dyaw
            (-10*deg_to_rad, 10*deg_to_rad),  # dpitch
            (-10*deg_to_rad, 10*deg_to_rad),  # droll
            (5, 20),  # thrust
        ])
    
    # Set up constraints
    print("Setting up constraints...")
    constraints = [
        {'type': 'eq', 'fun': lambda z: dynamics_constraints_robust(z, N, dt)},
        {'type': 'eq', 'fun': lambda z: boundary_constraints_robust(z, N, start_pose, goal_position, hoops)}
    ]

    if len(obstacles) > 0 and collision_constraints_optimized is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda z: -collision_constraints_optimized(z, N, obstacles, subsample=2)
        })
        print(f"  Added collision avoidance for {len(obstacles)} obstacles")
    elif len(obstacles) > 0 and collision_constraints_optimized is None:
        print(f"  ⚠ Warning: {len(obstacles)} obstacles present but collision_constraints_optimized not provided")
    
    # Start timing for leaderboard
    import time
    start_time = time.time()

    # Optimize with SLSQP
    print("\nOptimizing with SLSQP...\n")
    result = minimize(
        lambda z: cost_function_tuned(z, N, weights),
        z_init,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 400, 'ftol': 1e-5, 'disp': True}
    )

    # Fallback: try trust-constr if SLSQP fails
    if not result.success and result.status != 0:
        print("\n⚠ SLSQP did not fully converge, trying trust-constr...\n")

        # Convert to new-style constraints for trust-constr
        from scipy.optimize import NonlinearConstraint
        constraints_new = [
            NonlinearConstraint(
                lambda z: dynamics_constraints_robust(z, N, dt),
                lb=0, ub=0, keep_feasible=False
            ),
            NonlinearConstraint(
                lambda z: boundary_constraints_robust(z, N, start_pose, goal_position, hoops),
                lb=0, ub=0, keep_feasible=False
            )
        ]
        if len(obstacles) > 0 and collision_constraints_optimized is not None:
            constraints_new.append(
                NonlinearConstraint(
                    lambda z: collision_constraints_optimized(z, N, obstacles, subsample=2),
                    lb=-np.inf, ub=0, keep_feasible=False
                )
            )

        result = minimize(
            lambda z: cost_function_tuned(z, N, weights),
            result.x,  # Warm start from SLSQP result
            method='trust-constr',
            bounds=bounds,
            constraints=constraints_new,
            options={'maxiter': 80, 'verbose': 2}
        )

    # End timing for leaderboard
    end_time = time.time()
    optimization_time = end_time - start_time

    # Evaluate result
    print(f"\n{'='*60}")
    if result.success or result.status == 0:
        print("✅ OPTIMIZATION SUCCEEDED")
    else:
        print("⚠ OPTIMIZATION DID NOT FULLY CONVERGE")
    print(f"{'='*60}")
    print(f"Status: {result.message}")
    print(f"Iterations: {result.nit}")
    print(f"Final cost: {result.fun:.4f}")
    print(f"Optimization time: {optimization_time:.2f}s")
    
    # Compute violations
    try:
        dyn_viol = dynamics_constraints_robust(result.x, N, dt)
        bnd_viol = boundary_constraints_robust(result.x, N, start_pose, goal_position, hoops)
        max_dyn = np.max(np.abs(dyn_viol))
        max_bnd = np.max(np.abs(bnd_viol))
        max_violation = max(max_dyn, max_bnd)
        print(f"Max constraint violation: {max_violation:.6f}")
    except:
        max_violation = 1.0
        print(f"Could not compute violations")
    
    print(f"{'='*60}\n")
    
    # Extract trajectory with acceptance criteria matching documentation
    accept_solution = (max_violation < 0.01) or (result.success and max_violation < 0.1 and result.fun < 100)

    if accept_solution:
        states_opt, controls_opt = unpack_decision_vars(result.x, N)
        optimized_path = convert_states_to_poses(states_opt)
        info = {
            'success': True,
            'cost': result.fun,
            'iterations': result.nit,
            'max_violation': max_violation,
            'states': states_opt,
            'controls': controls_opt,
            'optimization_time': optimization_time
        }
        return optimized_path, True, info
    else:
        print("⛔ Constraints not satisfied, returning RRT path")
        info = {
            'success': False,
            'max_violation': max_violation,
            'cost': result.fun if hasattr(result, 'fun') else None,
            'optimization_time': optimization_time
        }
        return rrt_path, False, info


def optimize_racing_path_sequential(
    rrt_path: List[gtsam.Pose3],
    start_pose: gtsam.Pose3,
    hoops: List[gtsam.Pose3],
    obstacles: List,
    N: int = 25,
    dt: float = 0.1,
    weights: dict = None,
    # Student-implemented functions passed as parameters
    initialize_from_rrt_robust=None,
    dynamics_constraints_robust=None,
    boundary_constraints_robust=None,
    collision_constraints_optimized=None,
    cost_function_tuned=None,
    convert_states_to_poses=convert_states_to_poses,
    unpack_decision_vars=None
) -> Tuple[List[gtsam.Pose3], bool, dict]:
    """
    Optimize racing path sequentially (PROVIDED FUNCTION).

    Splits the full racing path into segments (one per hoop) and optimizes each.
    This mimics how drone_racing_rrt() plans incrementally.

    Arguments:
        rrt_path: Full RRT path through all hoops
        start_pose: Starting pose
        hoops: List of hoop poses
        obstacles: List of obstacles
        N: Knot points per segment
        dt: Time step duration
        weights: Cost function weights

        # Student-implemented functions (required):
        initialize_from_rrt_robust: Function to initialize decision variables from RRT
        dynamics_constraints_robust: Function enforcing dynamics constraints
        boundary_constraints_robust: Function enforcing boundary constraints
        collision_constraints_optimized: Function enforcing collision constraints
        cost_function_tuned: Combined cost function
        convert_states_to_poses: Function to convert states to poses
        unpack_decision_vars: Function to unpack decision variables

    Returns:
        optimized_path: Concatenated optimized segments
        success: Whether all segments succeeded
        info: Diagnostic information
    """
    if weights is None:
        weights = {'thrust': 0.05, 'angular': 0.5, 'smoothness': 5.0}
    
    print(f"\n{'='*80}")
    print(f"SEQUENTIAL RACING PATH OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Hoops: {len(hoops)}, Knot points per segment: {N}")
    print(f"{'='*80}\n")
    
    # Find hoop indices in RRT path
    hoop_indices = []
    for hoop in hoops:
        hoop_pos = hoop.translation()
        distances = [np.linalg.norm(p.translation() - hoop_pos) for p in rrt_path]
        hoop_indices.append(int(np.argmin(distances)))
    
    print("Splitting path into segments:")
    for i, idx in enumerate(hoop_indices):
        print(f"  Hoop {i+1} at RRT waypoint {idx}")
    
    # Create segments
    segments = []
    segment_starts = [0] + hoop_indices[:-1]
    segment_ends = hoop_indices
    
    for i in range(len(hoops)):
        start_idx = segment_starts[i]
        end_idx = segment_ends[i]
        segments.append(rrt_path[start_idx:end_idx+1])
    
    # Optimize each segment
    optimized_segments = []
    all_success = True
    current_start = start_pose
    
    for seg_idx, segment_rrt in enumerate(segments):
        print(f"\n{'---'*20}")
        print(f"SEGMENT {seg_idx+1}/{len(hoops)} → Hoop {seg_idx+1}")
        print(f"{'---'*20}\n")
        
        goal_pos = hoops[seg_idx].translation()
        
        opt_seg, success_seg, _ = optimize_trajectory(
            rrt_path=segment_rrt,
            start_pose=current_start,
            goal_position=goal_pos,
            hoops=[],
            obstacles=obstacles,
            N=N,
            dt=dt,
            weights=weights,
            # Pass student functions through
            initialize_from_rrt_robust=initialize_from_rrt_robust,
            dynamics_constraints_robust=dynamics_constraints_robust,
            boundary_constraints_robust=boundary_constraints_robust,
            collision_constraints_optimized=collision_constraints_optimized,
            cost_function_tuned=cost_function_tuned,
            convert_states_to_poses=convert_states_to_poses,
            unpack_decision_vars=unpack_decision_vars
        )
        
        if success_seg:
            print(f"✅ Segment {seg_idx+1} succeeded")
            optimized_segments.append(opt_seg)
            current_start = opt_seg[-1]
        else:
            print(f"❌ Segment {seg_idx+1} failed, using RRT")
            optimized_segments.append(opt_seg)
            all_success = False
            current_start = opt_seg[-1]
    
    # Concatenate
    print(f"\n{'='*80}")
    print("Concatenating segments...")
    optimized_path = []
    for i, seg in enumerate(optimized_segments):
        if i == 0:
            optimized_path.extend(seg)
        else:
            optimized_path.extend(seg[1:])
    
    print(f"Total optimized path: {len(optimized_path)} waypoints")
    print(f"Overall success: {all_success}")
    print(f"{'='*80}\n")
    
    info = {'success': all_success, 'segments': len(hoops)}
    return optimized_path, all_success, info


def visualize_rrt_vs_optimized_comparison(rrt_path, optimized_path, start_pose, goal_pose, title="RRT vs Optimized"):
    """Visualize RRT vs optimized paths side by side"""
    fig = go.Figure()
    
    # Start
    fig.add_trace(go.Scatter3d(
        x=[start_pose.x()], y=[start_pose.y()], z=[start_pose.z()],
        mode='markers', marker=dict(size=10, color='yellow'), name='Start'
    ))
    
    # Goal
    fig.add_trace(go.Scatter3d(
        x=[goal_pose.x()], y=[goal_pose.y()], z=[goal_pose.z()],
        mode='markers', marker=dict(size=10, color='green'), name='Goal'
    ))
    
    # RRT path
    rrt_x = [p.translation()[0] for p in rrt_path]
    rrt_y = [p.translation()[1] for p in rrt_path]
    rrt_z = [p.translation()[2] for p in rrt_path]
    fig.add_trace(go.Scatter3d(
        x=rrt_x, y=rrt_y, z=rrt_z,
        mode='lines+markers',
        line=dict(color='cyan', width=3),
        marker=dict(size=3),
        name='RRT Path'
    ))
    
    # Optimized path
    opt_x = [p.translation()[0] for p in optimized_path]
    opt_y = [p.translation()[1] for p in optimized_path]
    opt_z = [p.translation()[2] for p in optimized_path]
    fig.add_trace(go.Scatter3d(
        x=opt_x, y=opt_y, z=opt_z,
        mode='lines+markers',
        line=dict(color='magenta', width=4),
        marker=dict(size=4),
        name='Optimized Path'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[0, 10]),
            yaxis=dict(range=[0, 10]),
            zaxis=dict(range=[0, 10]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        width=1200,
        height=800
    )
    
    return fig


def drone_racing_path_comparison(hoops, start, rrt_path, optimized_path, title="RRT vs Optimized Racing"):
    """Compare RRT vs optimized racing paths"""
    fig = go.Figure()
    
    # Start
    fig.add_trace(go.Scatter3d(
        x=[start.x()], y=[start.y()], z=[start.z()],
        mode='markers', marker=dict(size=10, color='yellow'), name='Start'
    ))
    
    # Hoops
    for i, hoop in enumerate(hoops):
        xcrd, ycrd, zcrd = draw_hoop(hoop)
        fig.add_trace(go.Scatter3d(
            x=xcrd, y=ycrd, z=zcrd,
            mode='lines',
            line=dict(color='black', width=5),
            name=f'Hoop {i+1}' if i == 0 else None,
            showlegend=(i==0)
        ))
    
    # RRT path
    rrt_x = [p.translation()[0] for p in rrt_path]
    rrt_y = [p.translation()[1] for p in rrt_path]
    rrt_z = [p.translation()[2] for p in rrt_path]
    fig.add_trace(go.Scatter3d(
        x=rrt_x, y=rrt_y, z=rrt_z,
        mode='lines',
        line=dict(color='cyan', width=3),
        name='RRT Path'
    ))
    
    # Optimized path
    opt_x = [p.translation()[0] for p in optimized_path]
    opt_y = [p.translation()[1] for p in optimized_path]
    opt_z = [p.translation()[2] for p in optimized_path]
    fig.add_trace(go.Scatter3d(
        x=opt_x, y=opt_y, z=opt_z,
        mode='lines',
        line=dict(color='magenta', width=4),
        name='Optimized Path'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[0, 15]),
            yaxis=dict(range=[0, 15]),
            zaxis=dict(range=[0, 15]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        width=1400,
        height=900
    )
    
    return fig


def plot_velocity_acceleration_profiles(velocities, accelerations, controls, dt):
    """Plot velocity, acceleration, and thrust profiles"""
    from plotly.subplots import make_subplots
    
    N = len(velocities)
    time = np.arange(N) * dt
    speeds = np.linalg.norm(velocities, axis=1)
    accel_mags = np.linalg.norm(accelerations, axis=1)
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Velocity Profile', 'Acceleration Profile', 'Thrust Profile'),
        vertical_spacing=0.1
    )
    
    # Velocity
    fig.add_trace(go.Scatter(x=time, y=speeds, mode='lines', name='Speed', line=dict(color='blue', width=2)), row=1, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
    
    # Acceleration
    fig.add_trace(go.Scatter(x=time[:-1], y=accel_mags, mode='lines', name='Acceleration', line=dict(color='red', width=2)), row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=1)
    
    # Thrust
    fig.add_trace(go.Scatter(x=time, y=controls[:, 3], mode='lines', name='Thrust', line=dict(color='green', width=2)), row=3, col=1)
    fig.add_hline(y=10.0, line_dash="dash", line_color="gray", annotation_text="Hover", row=3, col=1)
    fig.update_yaxes(title_text="Thrust (N)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    
    fig.update_layout(height=900, width=1200, showlegend=False, title="Trajectory Profiles")
    return fig
# ============================================================================
# LEADERBOARD SYSTEM - Hyperparameter Tuning Competition
# ============================================================================

def compute_trajectory_jerk(positions: np.ndarray, dt: float = 0.1) -> float:
    """
    Compute average jerk magnitude from trajectory positions.

    Jerk = third derivative of position = d³p/dt³

    Args:
        positions: (N+1, 3) array of [x, y, z] positions at each time step
        dt: Time step duration (seconds)

    Returns:
        Average jerk magnitude (m/s³)

    Mathematical Background:
        velocity = dp/dt
        acceleration = d²p/dt²
        jerk = d³p/dt³

    Lower jerk = smoother trajectory = better for mechanical systems
    """
    # Compute derivatives using numerical gradient
    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)

    # Compute magnitude at each time step
    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    # Return average jerk
    return np.mean(jerk_magnitude)


def validate_hyperparams(N: int, weights: dict) -> Tuple[bool, str]:
    """
    Validate hyperparameters are within allowed ranges.

    Args:
        N: Number of knot points
        weights: Dict with 'thrust', 'angular', 'smoothness' keys

    Returns:
        (is_valid, error_message)

    Allowed Ranges:
        N ∈ [15, 50]
        w_thrust ∈ [0.01, 1.0]
        w_angular ∈ [0.1, 10.0]
        w_smoothness ∈ [1.0, 20.0]
    """
    # Validate N
    if not isinstance(N, int):
        return False, f"N must be int, got {type(N)}"
    if not (15 <= N <= 50):
        return False, f"N={N} out of range [15, 50]"

    # Validate weights dict
    required_keys = {'thrust', 'angular', 'smoothness'}
    if not all(k in weights for k in required_keys):
        return False, f"weights dict must have keys: {required_keys}"

    # Validate weight ranges
    w_t = weights['thrust']
    w_a = weights['angular']
    w_s = weights['smoothness']

    if not (0.01 <= w_t <= 1.0):
        return False, f"w_thrust={w_t:.3f} out of range [0.01, 1.0]"
    if not (0.1 <= w_a <= 10.0):
        return False, f"w_angular={w_a:.3f} out of range [0.1, 10.0]"
    if not (1.0 <= w_s <= 20.0):
        return False, f"w_smoothness={w_s:.3f} out of range [1.0, 20.0]"

    return True, "Valid hyperparameters"


def compute_leaderboard_score(
    optimized_path: List[gtsam.Pose3],
    info: dict,
    hyperparams: dict,
    reference_stats: dict = None
) -> dict:
    """
    Compute leaderboard score for hyperparameter tuning competition.

    Scoring Breakdown (100 points):
        1. Cost Quality (35 pts) - Lower optimized cost is better
        2. Constraint Satisfaction (25 pts) - Tighter constraint violations
        3. Smoothness (15 pts) - Lower trajectory jerk
        4. N Efficiency (10 pts) - Penalizes high knot points
        5. Time Efficiency (15 pts) - Penalizes slow optimization

    Args:
        optimized_path: Final optimized trajectory (List of Pose3)
        info: Optimization result dict with:
            - 'cost': final cost value
            - 'max_violation': maximum constraint violation
            - 'states': (N+1, 6) state array
            - 'optimization_time': time in seconds
        hyperparams: Dict with:
            - 'N': number of knot points
            - 'weights': {'thrust': ..., 'angular': ..., 'smoothness': ...}
        reference_stats: Optional dict with 'best_cost', 'worst_cost' for normalization
            (provided by Gradescope autograder for class-wide normalization)

    Returns:
        Dict with:
            - 'total_score': float (0-100)
            - 'cost_score': float (0-35)
            - 'constraint_score': float (0-25)
            - 'smoothness_score': float (0-15)
            - 'n_efficiency_score': float (0-10)
            - 'time_efficiency_score': float (0-15)
            - 'final_cost': float
            - 'max_violation': float
            - 'trajectory_jerk': float
            - 'N': int
            - 'optimization_time': float
            - 'weights': dict
            - 'breakdown': str (human-readable explanation)
    """
    # Validate inputs
    if not info.get('success', False):
        return {
            'total_score': 0.0,
            'error': 'Optimization failed',
            'breakdown': 'Score = 0 because optimization did not converge'
        }

    # Extract data
    N = hyperparams['N']
    weights = hyperparams['weights']
    final_cost = info['cost']
    max_violation = info['max_violation']
    opt_time = info['optimization_time']

    # Check timeout
    if opt_time > 600:  # 10 minute hard limit
        return {
            'total_score': 0.0,
            'error': 'Optimization exceeded 10 minute timeout',
            'optimization_time': opt_time,
            'breakdown': f'Score = 0 because optimization took {opt_time:.1f}s (> 600s limit)'
        }

    # === COMPONENT 1: Cost Quality (35 pts) ===
    if reference_stats and 'best_cost' in reference_stats and 'worst_cost' in reference_stats:
        # Normalize using class-wide stats (Gradescope provides this)
        best_cost = reference_stats['best_cost']
        worst_cost = reference_stats['worst_cost']
        if worst_cost > best_cost:
            cost_score = 35 * (1 - (final_cost - best_cost) / (worst_cost - best_cost))
            cost_score = np.clip(cost_score, 0, 35)
        else:
            cost_score = 35.0  # All costs are equal
    else:
        # Provisional score (will be normalized by Gradescope later)
        cost_score = 35.0

    # === COMPONENT 2: Constraint Satisfaction (25 pts) ===
    constraint_score = 25 * max(0, (0.1 - max_violation) / 0.1)
    constraint_score = np.clip(constraint_score, 0, 25)

    # === COMPONENT 3: Smoothness (15 pts) ===
    # Extract positions from states if available, otherwise from optimized_path
    if 'states' in info:
        positions = info['states'][:, :3]  # Extract x, y, z positions
    else:
        # Fallback: extract from optimized_path (List[gtsam.Pose3])
        positions = np.array([pose.translation() for pose in optimized_path])

    jerk = compute_trajectory_jerk(positions, dt=0.1)
    smoothness_score = 15 / (1 + jerk)
    smoothness_score = np.clip(smoothness_score, 0, 15)

    # === COMPONENT 4: N Efficiency (10 pts) ===
    n_efficiency_score = 10 * max(0, 1 - (N - 15) / 35)
    n_efficiency_score = np.clip(n_efficiency_score, 0, 10)

    # === COMPONENT 5: Time Efficiency (15 pts) ===
    time_efficiency_score = 15 * np.exp(-opt_time / 60)
    time_efficiency_score = np.clip(time_efficiency_score, 0, 15)

    # === TOTAL SCORE ===
    total_score = (cost_score + constraint_score + smoothness_score +
                   n_efficiency_score + time_efficiency_score)

    # Build breakdown explanation
    breakdown = f"""
        Leaderboard Score Breakdown:
        ════════════════════════════════════════════════════════════════
        Total Score:         {total_score:.2f} / 100

        Components:
        1. Cost Quality:              {cost_score:.2f} / 35
            Final cost: {final_cost:.2f}

        2. Constraint Satisfaction:   {constraint_score:.2f} / 25
            Max violation: {max_violation:.6f}

        3. Smoothness:                {smoothness_score:.2f} / 15
            Avg jerk: {jerk:.4f} m/s³

        4. N Efficiency:              {n_efficiency_score:.2f} / 10
            N = {N} (lower is better)

        5. Time Efficiency:           {time_efficiency_score:.2f} / 15
            Optimization time: {opt_time:.2f}s

        Hyperparameters:
        N = {N}
        Weights: thrust={weights['thrust']:.3f}, angular={weights['angular']:.3f}, smoothness={weights['smoothness']:.3f}
        ════════════════════════════════════════════════════════════════
        """

    return {
        'total_score': float(total_score),
        'cost_score': float(cost_score),
        'constraint_score': float(constraint_score),
        'smoothness_score': float(smoothness_score),
        'n_efficiency_score': float(n_efficiency_score),
        'time_efficiency_score': float(time_efficiency_score),
        'final_cost': float(final_cost),
        'max_violation': float(max_violation),
        'trajectory_jerk': float(jerk),
        'N': int(N),
        'optimization_time': float(opt_time),
        'weights': weights,
        'breakdown': breakdown
    }
