import datetime
import math
import numpy as np
import os
import pytz

from amr_planning.map import Map
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple, Set
import itertools


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
            action_costs: Cost of of moving one cell left, up, right, and down.

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
        try:
            r_goal, c_goal = self._xy_to_rc(goal)
            # if (r_goal > map_size[0] or r_goal < 0) or (c_goal > map_size[1] or c_goal < 0):
            if not self._map.contains(goal):
                raise ValueError(f"Goal: {goal} is outside the map!")
        except ValueError as e:
            print(e)

        try:
            r_start, c_start = self._xy_to_rc(start)
            # if (r_start > map_size[0] or r_start < 0) or (c_start > map_size[1] or c_start < 0):
            if not self._map.contains(start):
                raise ValueError(f"Start: {start} is outside the map!")
        except ValueError as e:
            print(e)

        path: List[Tuple[float, float]] = []
        steps: int = 0
        heuristic = self._compute_heuristic(goal)
        closed_list = set()
        ancestors: Dict[Tuple[int, int], Tuple[int, int]] = {}
        open_list: Dict[Tuple[int, int], Tuple[int, int]] = {
            (r_start, c_start): (heuristic[r_start, c_start], 0)
        }

        while open_list:
            node = r, c = min(open_list, key=lambda k: open_list.get(k)[0])
            _, g = open_list.pop(node)

            # if goal reached
            if node == (r_goal, c_goal):
                return self._reconstruct_path(start, goal, ancestors), steps

            neighbours = [(r, c - 1), (r - 1, c), (r, c + 1), (r + 1, c)]
            for i, neighbor in enumerate(neighbours):
                if (
                    self._map.contains(self._rc_to_xy(neighbor))
                    and neighbor not in open_list
                    and neighbor not in closed_list
                ):
                    g_new = g + self._action_costs[i]
                    f_new = g_new + heuristic[neighbor]
                    open_list[neighbor] = (f_new, g_new)
                    ancestors[neighbor] = node

            closed_list.add(node)
            steps += 1

        return path, steps

    @staticmethod
    def smooth_path(
        path, data_weight: float = 0.1, smooth_weight: float = 0.3, tolerance: float = 1e-9
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
        added_points = 2

        if added_points != 0:
            for _ in range(added_points):
                smoothed_path = []
                for i in range(len(path)):
                    if i == 0:
                        smoothed_path.append((float(path[i][0]), (float(path[i][1]))))
                        continue
                    n_x = (path[i][0] + path[i - 1][0]) / 2
                    n_y = (path[i][1] + path[i - 1][1]) / 2
                    smoothed_path.append((n_x, n_y))
                    smoothed_path.append((float(path[i][0]), (float(path[i][1]))))
                path[:] = smoothed_path[:]
        else:
            for i in range(len(path)):
                smoothed_path.append((float(path[i][0]), (float(path[i][1]))))

        change = float("inf")
        while change > tolerance:
            change = 0
            for i, s in enumerate(smoothed_path):
                if i == 0 or i == (len(smoothed_path) - 1):
                    continue
                p = path[i]
                s_x = (
                    s[0]
                    + data_weight * (p[0] - s[0])
                    + smooth_weight * (smoothed_path[i + 1][0] + smoothed_path[i - 1][0] - 2 * s[0])
                )
                s_y = (
                    s[1]
                    + data_weight * (p[1] - s[1])
                    + smooth_weight * (smoothed_path[i + 1][1] + smoothed_path[i - 1][1] - 2 * s[1])
                )
                smoothed_path[i] = (s_x, s_y)
                change += abs(s_x - s[0]) + abs(s_y - s[1])

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
        r_goal, c_goal = self._xy_to_rc(goal)
        # TODO: 3.1. Complete the missing function body with your code.
        map_size = np.shape(heuristic)
        for i, j in itertools.product(range(map_size[0]), range(map_size[1])):
            heuristic[i, j] = np.abs(r_goal - i) + np.abs(c_goal - j)

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
        path: List[Tuple[float, float]] = [goal]

        # TODO: 3.3. Complete the missing function body with your code.
        node = self._xy_to_rc(goal)
        while node:
            previous_node = ancestors[node]
            path.append(self._rc_to_xy(previous_node))
            node = previous_node
            if self._rc_to_xy(node) == (int(start[0]), int(start[1])):
                break
        path.reverse()
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
