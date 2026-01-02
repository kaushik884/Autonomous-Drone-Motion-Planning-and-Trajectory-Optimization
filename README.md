# Autonomous Drone Motion Planning and Trajectory Optimization

A complete drone racing motion planning system using RRT (Rapidly-exploring Random Trees) for path planning and nonlinear trajectory optimization for smooth, dynamically-feasible trajectories.

## Overview

This project implements an autonomous drone racing pipeline that:

1. **RRT Path Planning** - Generates collision-free paths through a racing course with hoops and obstacles
2. **Trajectory Optimization** - Refines RRT paths into smooth, dynamically-feasible trajectories
3. **Quadrotor Dynamics** - Models realistic drone dynamics including thrust, attitude control, and drag
4. **Obstacle Avoidance** - Handles spherical and box obstacles with safety margins
5. **Visualization** - Supports both Plotly (interactive 3D) and RViz2 (ROS 2)

## Features

- 6-DOF drone state representation (x, y, z, yaw, pitch, roll)
- Terminal velocity-based motion model
- Sequential racing through multiple hoops
- Collision constraints with configurable safety margins
- Cost function balancing thrust, angular velocity, and smoothness

## Project Structure

```
├── optimize.py              # Main script - runs full pipeline
├── rrt.py                   # RRT algorithm implementation
├── drone_dynamics.py        # Quadrotor dynamics model
├── cost_function.py         # Optimization cost functions
├── boundary_constraints.py  # Dynamics and boundary constraints
├── helpers_obstacles.py     # Obstacles, collision checking, visualization
├── visualize.py             # Visualization demos
└── tests.py                 # Unit tests
```

## Installation

### Prerequisites

- Ubuntu 22.04
- Python 3.10
- Conda (optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/kaushik884/Autonomous-Drone-Motion-Planning-and-Trajectory-Optimization.git
cd Autonomous-Drone-Motion-Planning-and-Trajectory-Optimization

# Create conda environment (recommended)
conda create -n drone_planning python=3.10 -y
conda activate drone_planning

# Install dependencies
pip install numpy scipy matplotlib plotly pandas
conda install -c conda-forge gtsam
```

## How to Run

### Quick Start

```bash
# Activate environment
conda activate drone_planning

# Run the main pipeline
python optimize.py
```

This will:
1. Generate an RRT path through the racing course
2. Optimize the trajectory for smoothness and dynamic feasibility
3. Open an interactive 3D visualization in your browser

### Expected Output

```
FINAL CHALLENGE: RACING WITH OBSTACLES (EASY)
================================================================

Configuration:
   - Start: [1, 3, 8]
   - Hoops: 4
   - Obstacles: 2 (EASY)

STAGE 1: RRT WITH OBSTACLES
----------------------------------------------------------------
RRT completed in 2.34s
Path: 156 waypoints
RRT is Collision-free!

STAGE 2: TRAJECTORY OPTIMIZATION
----------------------------------------------------------------
✅ OPTIMIZATION SUCCESS!
   Time: 8.21s

STAGE 3: VISUALIZATION
----------------------------------------------------------------
PATH COMPARISON:
  RRT path length: 24.53 m
  Optimized path length: 18.21 m
  Reduction: 25.8%
```

### Visualization Controls (Plotly)

- **Rotate**: Click and drag
- **Zoom**: Scroll wheel  
- **Pan**: Right-click and drag
- **Play Animation**: Click the play button

## Configuration

### Tuning Parameters

In `optimize.py`, you can adjust:

```python
N = 25                    # Number of trajectory points (15-50)
dt = 0.1                  # Time step (seconds)
weights = {
    'thrust': 0.05,       # Thrust cost weight
    'angular': 0.5,       # Angular velocity cost weight
    'smoothness': 5.0     # Smoothness cost weight
}
```

### Obstacle Difficulty

```python
# Easy obstacles (2 spheres)
obstacles = helpers.get_obstacles_easy()

# Hard obstacles (6 spheres + 2 boxes)
obstacles = helpers.get_obstacles_hard()
```

## Optional: RViz2 Visualization

For ROS 2 visualization (requires ROS 2 Humble):

```bash
# Terminal 1: Start RViz2
rviz2 -d drone_racing.rviz 

# Terminal 2: Run optimization
python optimize.py
```