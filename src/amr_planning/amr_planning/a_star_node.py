import rclpy
from rclpy.node import Node

from amr_msgs.msg import PoseStamped as AmrPoseStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import os
from typing import List, Tuple

from amr_planning.map import Map
from amr_planning.a_star import AStar


class AStarNode(Node):
    def __init__(self):
        """A* node initializer."""
        super().__init__("a_star")

        # Parameters
        self.declare_parameter("goal", (0.0, 0.0))
        self._goal = tuple(
            self.get_parameter("goal").get_parameter_value().double_array_value.tolist()
        )

        self.declare_parameter("world", "project")
        world = self.get_parameter("world").get_parameter_value().string_value

        # Subscriptions
        self._subscription_pose = self.create_subscription(
            AmrPoseStamped, "pose", self._path_callback, 10
        )

        # TODO: 3.5. Create the /path publisher (Path message).

        # Constants
        SENSOR_RANGE = 1.0  # Ultrasonic sensor range [m]

        # Attribute and object initializations
        map_path = os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "maps", world + ".json")
        )
        self._localized = False
        self._planning = AStar(map_path, SENSOR_RANGE)

    def _path_callback(self, pose_msg: AmrPoseStamped):
        """Subscriber callback. Executes A* and publishes the smoothed path to the goal.

        Args:
            pose_msg: Message containing the robot pose estimate.

        """
        if pose_msg.localized and not self._localized:
            # Execute A*
            start = (pose_msg.pose.position.x, pose_msg.pose.position.y)
            path, steps = self._planning.a_star(start, self._goal)
            smoothed_path = AStar.smooth_path(path, data_weight=0.1, smooth_weight=0.2)

            self.get_logger().info(f"Path found in {steps} steps.")
            self._planning.show(path, smoothed_path, save_figure=True)

            self._publish_path(smoothed_path)

        self._localized = pose_msg.localized

    def _publish_path(self, path: List[Tuple[float, float]]) -> None:
        """Publishes the robot's path to the goal in a nav_msgs.msg.Path message.

        Args:
            path: Smoothed path (initial location first) in (x, y) format.

        """
        # TODO: 3.6. Complete the function body with your code (i.e., replace the pass statement).
        pass


def main(args=None):
    rclpy.init(args=args)

    a_star_node = AStarNode()

    try:
        rclpy.spin(a_star_node)
    except KeyboardInterrupt:
        pass

    a_star_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
