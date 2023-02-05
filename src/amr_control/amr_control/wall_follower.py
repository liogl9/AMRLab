from typing import List, Tuple
import time
from math import pi


class WallFollower:
    """Class to safely explore an environment (without crashing) when the pose is unknown."""

    def __init__(self, dt: float):
        """Wall following class initializer.

        Args:
            dt: Sampling period [s].

        """
        self._dt: float = dt
        self.d_sensors = 0.33 / 2
        self.w0: float = -0.05
        self.turning: bool = False
        self.tr_rgt: bool = False
        self.tr_lft: bool = False

        self.rad_turn = 0.0
        self.kp = 15
        self.kd = self.kp * 0.6 * self._dt
        self.wall_ref = 0.35
        self.last_error = 0.0
        self.control = 1

    def go_straight(
        self, z_us: List[float], lft_wall: bool, rgt_wall: bool, control: int
    ) -> Tuple[float, float]:
        v = 0.5
        if control == 0:
            if rgt_wall:
                error = self.wall_ref - z_us[7]
                w = self.kp * error
            elif lft_wall:
                error = self.wall_ref - z_us[0]
                w = -self.kp * error
            else:
                w = self.w0
        elif control == 1:
            if rgt_wall:
                error = self.wall_ref - z_us[7]
                error_dev = (error - self.last_error) / self._dt
                self.last_error = error
                w = self.kp * error + self.kd * error_dev
            elif lft_wall:
                error = self.wall_ref - z_us[0]
                error_dev = (error - self.last_error) / self._dt
                self.last_error = error
                w = -(self.kp * error + self.kd * error_dev)
            else:
                w = self.w0
        elif control == 2:
            if rgt_wall:
                error = self.wall_ref - z_us[7]
                error_dev = (error - self.last_error) / self._dt
                self.last_error = error
                w = self.kp * error + self.kd * error_dev
            elif lft_wall:
                error = self.wall_ref - z_us[0]
                error_dev = (error - self.last_error) / self._dt
                self.last_error = error
                w = -(self.kp * error + self.kd * error_dev)
            else:
                w = self.w0

        return v, w

    def turn(self, deg: float, z_w: float, on_point: bool) -> Tuple[float, float]:
        """
        turn _summary_

        _extended_summary_

        Args:
            deg (float): _description_

        Returns:
            Tuple[float, float]: _description_
        """
        if not on_point:
            v = 0.0
            inc_w = 1
        else:
            v = 0.0
            inc_w = 1

        if self.rad_turn < (abs(deg) * pi / 180.0):
            self.rad_turn = self.rad_turn + abs(z_w) * self._dt
            if deg > 0.0:
                w = self.w0 + inc_w
            else:
                w = self.w0 - inc_w
        else:
            self.rad_turn = 0.0
            self.turning = False
            self.tr_lft = False
            self.tr_rgt = False
            v = 0.0
            w = 0.0

        return v, w

    def get_sides(self, z_us: List[float]) -> Tuple[bool, bool, bool, bool]:
        """
        get_sides _summary_

        _extended_summary_

        Args:
            z_us (List[float]): _description_

        Returns:
            Tuple[bool, bool]: True if it detects a wall
        """
        left_sns = [z_us[0], z_us[-1]]
        right_sns = [z_us[7], z_us[8]]
        front_sns = [z_us[3], z_us[4]]
        rear_sns = [z_us[11], z_us[12]]

        left = all([x < 0.9 for x in left_sns])
        right = all([x < 0.9 for x in right_sns])
        front = all([x < 0.35 for x in front_sns])
        rear = all([x < 0.35 for x in rear_sns])

        return left, right, front, rear

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
        # Control 1: Pasillo ida y vuelta
        # Check walls
        left, right, front, rear = self.get_sides(z_us)

        if not front and not self.turning:
            v, w = self.go_straight(z_us, left, right, self.control)
        else:
            if not self.turning:
                self.turning = True
                if right and left:
                    self.tr_rgt = True
                    self.tr_lft = True
                elif right:
                    self.tr_rgt = True
                    self.tr_lft = False
                elif left:
                    self.tr_lft = True
                    self.tr_rgt = False
                elif not (right or left):
                    self.tr_lft = True
                    self.tr_rgt = False
            if self.tr_rgt and self.tr_lft:
                v, w = self.turn(180, z_w, True)
            elif self.tr_rgt:
                v, w = self.turn(90, z_w, False)
            elif self.tr_lft:
                v, w = self.turn(-90, z_w, False)
            else:
                v = 0.0
                w = 0.0

        # front_sns_cls = [x < 0.4 for x in z_us[2:6]]
        # if not (self.turn_left or self.turn_right):
        #     if all(front_sns_cls):
        #         v = 0.0
        #         w = 0.0
        #         self.turn = True
        #         left, right = self.get_sides(z_us)
        #         if not left:
        #             self.turn_left = True
        #         else:
        #             self.turn_right = True

        #     else:
        #         v, w = self.go_straight(z_us, z_v, z_w)
        # else:
        #     self.rad_turn = self.rad_turn + abs(z_w) * self._dt
        #     self.time = time.time()
        #     v = 0.0
        #     if self.rad_turn < 3.14:
        #         if self.turn_right:
        #             w = self.w0 - 0.4
        #         else:
        #             w = self.w0 + 0.4
        #     else:
        #         self.rad_turn = 0
        #         self.turn_right = False
        #         self.turn_left = False
        #         v = 0.0
        #         w = 0.0

        # w>0 izq; w<0 der
        # w = 0.0

        return v, w
