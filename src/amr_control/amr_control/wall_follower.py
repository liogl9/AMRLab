from typing import List, Tuple
from math import pi


class WallFollower:
    """Class to safely explore an environment (without crashing) when the pose is unknown."""

    def __init__(self, dt: float):
        """Wall following class initializer.

        Args:
            dt: Sampling period [s].

        """
        self._dt: float = dt  # Sampling time

        # Turn variable
        self.turning: bool = False  # Define generic state of robot turning
        self.turn_rgt: bool = False  # Define state of robot turning right
        self.turn_lft: bool = False  # efine state of robot turning left
        self.rad_turn: float = 0.0  # Cumulative radians turned

        # Control parameters
        self.kp: float = 7  # Proportional parameter
        self.kd: float = self.kp * 0.6 * self._dt  # Derivative parameter
        self.last_error: float = 0.0  # Last error measured
        self.w0: float = -0.05  # Base angular speed
        self.wall_ref: float = 0.35  # Wall distance reference
        self.control: int = 1  # Control type: 0 -> P
        #                                      1 -> PD
        # Thresholds for detecting a wall
        self.front_thr: float = 0.65  # Wall distance for front sensors
        self.side_thr: float = 0.9  # Wall distance for side sensors

    def go_straight(
        self, z_us: List[float], lft_wall: bool, rgt_wall: bool, control: int
    ) -> Tuple[float, float]:
        """
        go_straight: Function to make the robot follow a side wall (priorizing the right one)
        if detected or keep going straight with a small angular velocity to compensate any possible
        error if not

        Args:
            z_us (List[float]): List measure of the ultrasonic sensors
            lft_wall (bool): True if there is a near left wall
            rgt_wall (bool): False if there is a near left wall
            control (int): Control type to be used 0 -> P
                                                   1 -> PD
        Returns:
            v, w Tuple[float, float]: Command of the linear and angular velocities
        """
        v = 0.5  # Base linear velocity on straight corridors

        if rgt_wall:
            # Following right wall
            error = self.wall_ref - z_us[7]  # Measure error
            if control == 0:
                # P control
                w = self.kp * error  # Implement control
            elif control == 1:
                # PD control
                error_dev = (error - self.last_error) / self._dt  # Calculate derivative
                self.last_error = error  # Update last error
                w = self.kp * error + self.kd * error_dev  # Implement control
        elif lft_wall:
            # Following left wall
            error = self.wall_ref - z_us[0]  # Measure error
            if control == 0:
                # P control
                w = -self.kp * error  # Implement control
            elif control == 1:
                # PD control
                error_dev = (error - self.last_error) / self._dt
                self.last_error = error
                w = -(self.kp * error + self.kd * error_dev)
        else:
            # No walls
            w = self.w0  # Keep going with a small angular valocity to compensate

        return v, w

    def turn(self, deg: float, z_w: float, on_point: bool) -> Tuple[float, float]:
        """
        turn Function to make the robot turn the degrees given on its place or with a small linear
        velocity

        Args:
            deg (float): Degrees to be turned
            z_w (float): Actual angular speed
            on_point (bool): Wether to turn on itself (True) or moving forward too (False)

        Returns:
            v, w Tuple[float, float]: Command of the linear and angular velocities
        """
        # Define base linear velocity and angular velocity increments respect the base
        if not on_point:
            v = 0.0
            inc_w = 1.3
        else:
            v = 0.0
            inc_w = 1.4

        # Check if it needs to turn more or not
        if self.rad_turn < (abs(deg) * pi / 180.0):
            self.rad_turn = self.rad_turn + abs(z_w) * self._dt
            if deg > 0.0:
                w = self.w0 + inc_w
            else:
                w = self.w0 - inc_w
        else:
            # If it has turned the specified degrees go back to base state
            self.rad_turn = 0.0
            self.turning = False
            self.turn_lft = False
            self.turn_rgt = False
            v = 0.0
            w = 0.0

        return v, w

    def choose_turn(self, right: bool, left: bool):
        """
        choose_turn: Function to specify where does the robot have to turn

        Args:
            right (bool): Wether if there is a near wall on the right (True) or not (False)
            left (bool): Wether if there is a near wall on the left (True) or not (False)
        """
        if right and left:
            self.turn_rgt = True
            self.turn_lft = True
        elif right:
            self.turn_rgt = False
            self.turn_lft = True
        elif left:
            self.turn_lft = False
            self.turn_rgt = True
        elif not (right or left):
            self.turn_lft = False
            self.turn_rgt = True

    def get_sides(self, z_us: List[float]) -> Tuple[bool, bool, bool, bool]:
        """
        get_sides: Check if ther is any wall on the front, the rear or the sides

        Args:
            z_us (List[float]): List measure of the ultrasonic sensors

        Returns:
            left, right, front, rear Tuple[bool, bool, bool, bool]: True if it detects a wall
        """
        left_sns = [z_us[0], z_us[-1]]
        right_sns = [z_us[7], z_us[8]]
        front_sns = [z_us[3], z_us[4]]
        rear_sns = [z_us[11], z_us[12]]

        left = all([x < self.side_thr for x in left_sns])
        right = all([x < self.side_thr for x in right_sns])
        front = all([x < self.front_thr for x in front_sns])
        rear = all([x < 0.65 for x in rear_sns])

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
            # If there is no wall on the front side and it is not turning go straight
            v, w = self.go_straight(z_us, left, right, self.control)
        else:
            # If there is a wall on the front or the robot is turning
            if not self.turning:
                # If it wasn't turning but there is a wall on the front
                self.turning = True  # start the turning state
                self.choose_turn(right, left)  # define how much to turn and the orientation
            if self.turn_rgt and self.turn_lft:
                # If there are walls on both sides
                v, w = self.turn(170, z_w, True)
            elif self.turn_lft:
                # If there are walls on the right
                v, w = self.turn(70, z_w, False)
            elif self.turn_rgt:
                # If there are walls on the left
                v, w = self.turn(-70, z_w, False)
            else:
                v = 0.0
                w = 0.0

        # w>0 izq; w<0 der

        return v, w
