import math

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    world = "project"
    start = (2.0, -3.0, 1.5 * math.pi)
    goal = (3.0, 2.0)
    particles = 6500

    return LaunchDescription(
        [
            Node(
                package="amr_control",
                executable="wall_follower",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
                parameters=[{"enable_localization": True}],
            ),
            Node(
                package="amr_localization",
                executable="particle_filter",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
                parameters=[{"particles": particles, "world": world}],
            ),
            Node(
                package="amr_planning",
                executable="a_star",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
                parameters=[{"goal": goal, "world": world}],
            ),
            Node(
                package="amr_control",
                executable="pure_pursuit",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
            ),
            Node(
                package="amr_simulation",
                executable="coppeliasim",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
                parameters=[{"enable_localization": True, "start": start, "goal": goal}],
            ),  # Must be launched last
        ]
    )
