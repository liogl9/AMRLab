import math

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    world = "lab02"
    start = (0, -1.5, 0.5 * math.pi)
    particles = 4000

    return LaunchDescription(
        [
            Node(
                package="amr_control",
                executable="wall_follower",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
            ),
            Node(
                package="amr_localization",
                executable="particle_filter",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
                parameters=[{"enable_plot": True, "particles": particles, "world": world}],
            ),
            Node(
                package="amr_simulation",
                executable="coppeliasim",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
                parameters=[{"enable_localization": True, "start": start}],
            ),  # Must be launched last
        ]
    )
