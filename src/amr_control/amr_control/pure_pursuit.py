from typing import List, Tuple
import numpy as np


class PurePursuit:
    """Class to follow a path using a simple pure pursuit controller."""

    def __init__(self, dt: float, lookahead_distance: float = 0.5):
        """Pure pursuit class initializer.

        Args:
            dt: Sampling period [s].
            lookahead_distance: Distance to the next target point [m].

        """
        self._dt: float = dt
        self._lookahead_distance: float = lookahead_distance
        self._path: List[Tuple[float, float]] = []
        self._remaining_path: List[Tuple[float, float]] = []
        self.alfa = 0

    def compute_commands(self, x: float, y: float, theta: float) -> Tuple[float, float]:
        """Pure pursuit controller implementation.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].
            theta: Estimated robot heading [rad].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        # TODO: 4.4. Complete the function body with your code (i.e., compute v and w).
        v = 0.0
        w = 0.0

        if len(self.path):
            origin, origin_idx = self._find_closest_point(x, y)
            target = self._find_target_point(origin, origin_idx)
            self.alfa = theta - np.arctan2(target[1] - y, target[0] - x)
            if abs(self.alfa) > 0.2:
                v = 0.5
            else:
                v = 0.9
            w = 2 * v * np.sin(self.alfa) / self._lookahead_distance * -1.0

        return v, w

    @property
    def path(self) -> List[Tuple[float, float]]:
        """Path getter."""
        return self._path

    @path.setter
    def path(self, value: List[Tuple[float, float]]) -> None:
        """Path setter."""
        self._path = value

    def _find_closest_point(self, x: float, y: float) -> Tuple[Tuple[float, float], int]:
        """Find the closest path point to the current robot pose.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].

        Returns:
            Tuple[float, float]: (x, y) coordinates of the closest path point [m].
            int: Index of the path point found.

        """
        # TODO: 4.2. Complete the function body (i.e., find closest_xy and closest_idx).
        closest_xy = (0.0, 0.0)
        closest_idx = 0
        distances = np.array(
            [np.sqrt((node[0] - x) ** 2 + (node[1] - y) ** 2) for node in self.path]
        )
        closest_idx = np.where(distances == distances.min())[0][0]
        closest_xy = self.path[closest_idx]
        self._remaining_path[:] = self.path[closest_idx:]

        return closest_xy, closest_idx

    def _find_target_point(
        self, origin_xy: Tuple[float, float], origin_idx: int
    ) -> Tuple[float, float]:
        """Find the destination path point based on the lookahead distance.

        Args:
            origin_xy: Current location of the robot (x, y) [m].
            origin_idx: Index of the current path point.

        Returns:
            Tuple[float, float]: (x, y) coordinates of the target point [m].

        """
        # TODO: 4.3. Complete the function body with your code (i.e., determine target_xy).
        target_xy = (0.0, 0.0)

        distances = np.array(
            [
                abs(
                    np.sqrt((node[0] - origin_xy[0]) ** 2 + (node[1] - origin_xy[1]) ** 2)
                    - self._lookahead_distance
                )
                for node in self._remaining_path
            ]
        )
        closest_idx = np.where(distances == distances.min())[0][0]
        target_xy = self._remaining_path[closest_idx]

        return target_xy