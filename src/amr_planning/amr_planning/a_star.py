import datetime
import math
import numpy as np
import os
import pytz

from amr_planning.map import Map
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple


class AStar:
    """Class to plan the optimal path to a given location using the A* algorithm."""

    def __init__(
        self,
        map_path: str,
        sensor_range: float,
        action_costs: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ):
        """A* class initializer.

        Args:
            map_path: Path to the map of the environment.
            sensor_range: Sensor measurement range [m].
            action_costs: Cost of of moving one cell left, right, up, and down.

        """
        self._actions: np.ndarray = np.array(
            [
                (-1, 0),  # Move one cell left
                (0, 1),  # Move one cell up
                (1, 0),  # Move one cell right
                (0, -1),  # Move one cell down
            ]
        )
        self._action_costs: Tuple[float, float, float, float] = action_costs
        self._map: Map = Map(map_path, sensor_range, compiled_intersect=False, use_regions=False)

        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    def a_star(
        self, start: Tuple[float, float], goal: Tuple[float, float]
    ) -> Tuple[List[Tuple[float, float]], int]:
        """Computes the optimal path to a given goal location using the A* algorithm.

        Args:
            start: Initial location in (x, y) format.
            goal: Destination in (x, y) format.

        Returns:
            Path to the destination. The first value corresponds to the initial location.
            Number of A* iterations required to find the path.

        """
        # TODO: 3.2. Complete the function body (i.e., replace the code below).
        path: List[Tuple[float, float]] = []
        steps: int = 0
        return path, steps

    @staticmethod
    def smooth_path(
        path, data_weight: float = 0.1, smooth_weight: float = 0.1, tolerance: float = 1e-6
    ) -> List[Tuple[float, float]]:
        """Computes a smooth trajectory from a Manhattan-like path.

        Args:
            path: Non-smoothed path to the goal (start location first).
            data_weight: The larger, the more similar the output will be to the original path.
            smooth_weight: The larger, the smoother the output path will be.
            tolerance: The algorithm will stop when after an iteration the smoothed path changes
                       less than this value.

        Returns: Smoothed path (initial location first) in (x, y) format.

        """
        smoothed_path: List[Tuple[float, float]] = []

        # TODO: 3.4. Complete the missing function body with your code.

        return smoothed_path

    @staticmethod
    def plot(axes, path: List[Tuple[float, float]], smoothed_path: List[Tuple[float, float]] = ()):
        """Draws a path.

        Args:
            axes: Figure axes.
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).

        Returns:
            axes: Modified axes.

        """
        x_val = [x[0] for x in path]
        y_val = [x[1] for x in path]

        axes.plot(x_val, y_val)  # Plot the path
        axes.plot(
            x_val[1:-1], y_val[1:-1], "bo", markersize=4
        )  # Draw blue circles in every intermediate cell

        if smoothed_path:
            x_val = [x[0] for x in smoothed_path]
            y_val = [x[1] for x in smoothed_path]

            axes.plot(x_val, y_val, "y")  # Plot the path
            axes.plot(
                x_val[1:-1], y_val[1:-1], "yo", markersize=4
            )  # Draw yellow circles in every intermediate cell

        axes.plot(x_val[0], y_val[0], "rs", markersize=7)  # Draw a red square at the start location
        axes.plot(
            x_val[-1], y_val[-1], "g*", markersize=12
        )  # Draw a green star at the goal location

        return axes

    def show(
        self,
        path,
        smoothed_path=(),
        title: str = "Path",
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays a given path on the map.

        Args:
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).
            title: Plot title.
            display: True to open a window to visualize the particle filter evolution in real-time.
                Time consuming. Does not work inside a container unless the screen is forwarded.
            block: True to stop program execution until the figure window is closed.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, path, smoothed_path)

        axes.set_title(title)
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait for 1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.join(os.path.dirname(__file__), "..", save_dir)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = f"{self._timestamp} {title.lower()}.png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _compute_heuristic(self, goal: Tuple[float, float]) -> np.ndarray:
        """Creates an admissible heuristic.

        Args:
            goal: Destination location in (x,y) coordinates.

        Returns:
            Admissible heuristic.

        """
        heuristic = np.zeros_like(self._map.grid_map)

        # TODO: 3.1. Complete the missing function body with your code.

        return heuristic

    def _reconstruct_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        ancestors: Dict[Tuple[int, int], Tuple[int, int]],
    ) -> List[Tuple[float, float]]:
        """Computes the path from the start to the goal given the ancestors of a search algorithm.

        Args:
            start: Initial location in (x, y) format.
            goal: Goal location in (x, y) format.
            ancestors: Matrix that contains for every cell, None or the (x, y) ancestor from which
                       it was opened.

        Returns: Path to the goal (start location first) in (x, y) format.

        """
        path: List[Tuple[float, float]] = []

        # TODO: 3.3. Complete the missing function body with your code.

        return path

    def _xy_to_rc(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """Converts (x, y) coordinates of a metric map to (row, col) coordinates of a grid map.

        Args:
            xy: (x, y) [m].

        Returns:
            rc: (row, col) starting from (0, 0) at the top left corner.

        """
        map_rows, map_cols = np.shape(self._map.grid_map)

        x = round(xy[0])
        y = round(xy[1])

        row = int(map_rows - (y + math.ceil(map_rows / 2.0)))
        col = int(x + math.floor(map_cols / 2.0))

        return row, col

    def _rc_to_xy(self, rc: Tuple[int, int]) -> Tuple[float, float]:
        """Converts (row, col) coordinates of a grid map to (x, y) coordinates of a metric map.

        Args:
            rc: (row, col) starting from (0, 0) at the top left corner.

        Returns:
            xy: (x, y) [m].

        """
        map_rows, map_cols = np.shape(self._map.grid_map)
        row, col = rc

        x = col - math.floor(map_cols / 2.0)
        y = map_rows - (row + math.ceil(map_rows / 2.0))

        return x, y
