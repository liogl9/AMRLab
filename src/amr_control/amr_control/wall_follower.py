from typing import List, Tuple


class WallFollower:
    """Class to safely explore an environment (without crashing) when the pose is unknown."""

    def __init__(self, dt: float):
        """Wall following class initializer.

        Args:
            dt: Sampling period [s].

        """
        self._dt: float = dt
        self._side_trh: float = 0.1
        self._w0: float = -1.0

    def compute_commands(self, z_us: List[float], z_v: float, z_w: float) -> Tuple[float, float]:
        """Wall following exploration algorithm.

        Args:
            z_us: Distance from every ultrasonic sensor to the closest obstacle [m].
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        # TODO: 1.14. Complete the function body with your code (i.e., compute v and w).
        # Control 0
        v = 0.5
        w = self._w0
        if z_us[0] < self._side_trh and z_us[-1] < self._side_trh:
            w = self._w0 - 0.5
        elif z_us[7] < self._side_trh and z_us[8] < self._side_trh:
            w = self._w0 + 0.2
        # w>0 izq; w<0 der
        # w = 0.0

        return v, w
