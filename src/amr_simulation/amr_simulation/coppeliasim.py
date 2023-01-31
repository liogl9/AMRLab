import math
import socket
import time

from amr_simulation.zmqRemoteApi import RemoteAPIClient
from typing import Tuple


class CoppeliaSim:
    """CoppeliaSim robot simulator driver."""

    def __init__(self, dt: float, start: Tuple[float, float, float], goal_tolerance: float):
        """CoppeliaSim driver initializer.

        Args:
            dt: Sampling period [s].
            start: Initial robot location (x, y, theta) [m, m, rad].
            goal_tolerance: Maximum distance between the robot center and the goal to succeed [m].

        """
        # Initialize attributes
        self._dt: float = dt
        self._goal_tolerance: float = goal_tolerance
        self._steps: int = 0

        self._client = RemoteAPIClient(
            host=socket.gethostbyname("host.docker.internal"), port=23000
        )

        self._client.setStepping(True)

        # Start simulation
        self._sim = self._client.getObject("sim")
        self._sim.stopSimulation()
        self._sim.startSimulation()
        self.next_step(increment_step=False)  # Wait for the simulation to start

        # Create the robot
        self._robot_handle = self._create_robot(start)
        self.next_step(increment_step=False)  # Wait for the robot to be created

        # Wait one second to allow other nodes to properly initialize
        time.sleep(1)

        self._start_time: float = time.time()

    @property
    def sim(self):
        """Simulation handle getter.

        Returns:
            An object that allows calling regular CoppeliaSim API functions
            (https://www.coppeliarobotics.com/helpFiles/en/apiFunctions.htm).

        """
        return self._sim

    def check_position(self, x: float, y: float) -> Tuple[Tuple[float, float, float], float, bool]:
        """_summary_

        Args:
            x: x world coordinate [m].
            y: y world coordinate [m].

        Returns:
            real_pose: Real robot pose (x, y, theta) [m, m, rad].
            position_error: Distance between the query point and the true robot location [m].
            within_tolerance: True if the robot center is within the goal tolerance radius.

        """
        real_position = self._sim.getObjectPosition(self._robot_handle, -1)
        real_orientation = self._sim.getObjectOrientation(self._robot_handle, -1)

        position_error = math.dist(real_position[0:2], (x, y))
        real_pose = (real_position[0], real_position[1], real_orientation[2])

        return real_pose, position_error, position_error <= self._goal_tolerance

    def next_step(self, increment_step: bool = True):
        """Advances the simulation time one sampling period (dt).

        Args:
            increment_step: False not to count an execution step (for initialization purposes).

        """
        self._client.step()
        self._steps += int(increment_step)

    def stop_simulation(self) -> Tuple[float, float, int]:
        """Finishes the simulation and computes execution statistics.

        Returns:
            execution_time: Natural (real) time since the start of the simulation.
            simulated_time: Time elapsed inside the simulation.
            steps: Number of sampling periods executed (simulated_time / dt).

        """
        # Compute statistics
        execution_time = time.time() - self._start_time
        simulated_time = self._steps * self._dt

        # Stop the simulation
        self._sim.stopSimulation()

        return execution_time, simulated_time, self._steps

    def _create_robot(self, pose: Tuple[float, float, float]):
        """Spawns a Pioneer 3-DX robot at a given pose.

        Args:
            pose: (x, y, theta) [m, m, rad] in world coordinates.

        Returns:
            Robot handle.

        """
        out_ints, _, _, _ = self._sim.callScriptFunction(
            "createRobot@/Maze",
            self._sim.scripttype_childscript,
            [],
            pose,
            ["p3dx.ttm"],
            "",
        )
        robot_handle = out_ints[0]

        return robot_handle
