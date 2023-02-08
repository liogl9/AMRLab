import math

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # start = (4.0, -4.0, 0.5 * math.pi)  # Outer corridor
    start = (2.0, -3.0, 1.5 * math.pi)  # Inner corridor

    return LaunchDescription(
        [
            Node(
                package="amr_control",
                executable="wall_follower",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
            ),
            Node(
                package="amr_simulation",
                executable="coppeliasim",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
                parameters=[{"start": start}],
            ),  # Must be launched last
        ]
    )
