import datetime
import math
import numpy as np
import os
import pytz

from amr_localization.map import Map
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from typing import List, Tuple


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        sensors: List[Tuple[float, float, float]],
        sensor_range: float,
        particle_count: int,
        sigma_v: float = 0.15,
        sigma_w: float = 0.75,
        sigma_z: float = 0.25,
    ):
        """Particle filter class initializer.

        Args:
            dt: Sampling period [s].
            map_path: Path to the map of the environment.
            sensors: Robot sensors' pose in the robot coordinate frame (x, y, theta) [m, m, rad].
            sensor_range: Sensor measurement range [m].
            particle_count: Initial number of particles.
            sigma_v: Standard deviation of the linear velocity [m/s].
            sigma_w: Standard deviation of the angular velocity [rad/s].
            sigma_z: Standard deviation of the measurements [m].

        """
        self._dt: float = dt
        self._initial_particle_count: int = particle_count
        self._particle_count: int = particle_count
        self._sensors: List[Tuple[float, float, float]] = sensors
        self._sensor_range: float = sensor_range
        self._sigma_v: float = sigma_v
        self._sigma_w: float = sigma_w
        self._sigma_z: float = sigma_z
        self._iteration: int = 0

        self._map = Map(map_path, sensor_range, compiled_intersect=True, use_regions=True)
        self._particles = self._init_particles(particle_count)
        self._ds, self._phi = self._init_sensor_polar_coordinates(sensors)
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    def compute_pose(self) -> Tuple[bool, Tuple[float, float, float]]:
        """Computes the pose estimate when the particles form a single DBSCAN cluster.

        Adapts the amount of particles depending on the number of clusters during localization.
        100 particles are kept for pose tracking.

        Returns:
            localized: True if the pose estimate is valid.
            pose: Robot pose estimate (x, y, theta) [m, m, rad].

        """
        localized: bool = False
        pose: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

        # TODO: 2.10. Complete the missing function body with your code.
        clustering = DBSCAN(eps=0.33, min_samples=20).fit(self._particles[:, :-1])
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        centroid = np.empty((1, 3))

        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = self._particles[:, :-1][class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = self._particles[:, :-1][class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title(f"Estimated number of clusters: {n_clusters}")
        plt.savefig("clusters.png")
        if n_clusters == 1:
            localized = True
            centroid[0, 0:2] = np.mean(self._particles[:, :-1], axis=0)
            # centroid[0, 2] = np.mean(
            #     [x if x <= math.pi else x - math.pi * 2 for x in self._particles[:, 2]]
            # )
            centroid[0, 2] = np.mean(self._particles[:, 2])
            if centroid[0, 2] < 0:
                centroid[0, 2] += math.pi * 2

            pose = (centroid[0, 0], centroid[0, 1], centroid[0, 2])
            self._particle_count = 100

        elif n_clusters > 1 and n_clusters <= 4:
            self._particle_count = 400 * n_clusters
            if self._particle_count < 400:
                self._particle_count = 400

        return localized, pose

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1

        # TODO: 2.5. Complete the function body with your code (i.e., replace the pass statement).
        for i in range(self._particle_count):
            v_noise = np.random.normal(v, self._sigma_v)
            w_noise = np.random.normal(w, self._sigma_w)
            particle = np.empty(2)

            particle[0] = (
                self._particles[i, 0] + v_noise * math.cos(self._particles[i, 2]) * self._dt
            )
            particle[1] = (
                self._particles[i, 1] + v_noise * math.sin(self._particles[i, 2]) * self._dt
            )
            self._particles[i, 2] = self._particles[i, 2] + w_noise * self._dt

            if self._particles[i, 2] > 2 * math.pi:
                self._particles[i, 2] = self._particles[i, 2] - 2 * math.pi
            elif self._particles[i, 2] < 0:
                self._particles[i, 2] = self._particles[i, 2] + 2 * math.pi

            intersection, _ = self._map.check_collision(
                [(self._particles[i, 0], self._particles[i, 1]), (particle[0], particle[1])]
            )
            if intersection:
                self._particles[i, 0] = intersection[0]
                self._particles[i, 1] = intersection[1]
            else:
                self._particles[i, 0] = particle[0]
                self._particles[i, 1] = particle[1]

    def resample(self, measurements: List[float]) -> None:
        """Samples a new set of particles.

        Args:
            measurements: Sensor measurements [m].

        """
        # TODO: 2.9. Complete the function body with your code (i.e., replace the pass statement).
        weights = [
            self._measurement_probability(measurements, particle) for particle in self._particles
        ]
        # for particle in self._particles:
        #     weights.append(self._measurement_probability(measurements, particle))
        weights_norm = weights / np.array(weights).sum(0)
        new_particle_idxs = np.random.choice(
            self._particles.shape[0], self._particle_count, replace=True, p=weights_norm
        )
        self._particles = self._particles[new_particle_idxs]

    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """
        if orientation:
            dx = [math.cos(particle[2]) for particle in self._particles]
            dy = [math.sin(particle[2]) for particle in self._particles]
            axes.quiver(
                self._particles[:, 0],
                self._particles[:, 1],
                dx,
                dy,
                color="b",
                scale=15,
                scale_units="inches",
            )
        else:
            axes.plot(self._particles[:, 0], self._particles[:, 1], "bo", markersize=1)

        return axes

    def show(
        self,
        title: str = "",
        orientation: bool = True,
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "img",
    ):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            orientation: Draw particle orientation.
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
        axes = self.plot(axes, orientation)

        axes.set_title(title + " (Iteration #" + str(self._iteration) + ")")
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait 1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", save_dir, self._timestamp)
            )

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = str(self._iteration).zfill(4) + " " + title.lower() + ".png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _init_particles(self, particle_count: int) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3*pi/2].

        Args:
            particle_count: Number of particles.

        Returns: A NumPy array of tuples (x, y, theta) [m, m, rad].

        """
        particles = np.empty((particle_count, 3), dtype=object)

        # TODO: 2.4. Complete the missing function body with your code.
        x_min, y_min, x_max, y_max = self._map.bounds()
        for i in range(particle_count):
            x = x_max + 1.0
            y = y_max + 1.0
            while not self._map.contains((x, y)):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)

            theta = np.random.choice([0, math.pi / 2, math.pi, 3 * math.pi / 2])
            particles[i] = np.array([x, y, theta])

        return particles

    @staticmethod
    def _init_sensor_polar_coordinates(
        sensors: List[Tuple[float, float, float]]
    ) -> Tuple[List[float], List[float]]:
        """Converts the sensors' poses to polar coordinates wrt to the robot's coordinate frame.

        Args:
            sensors: Robot sensors location and orientation (x, y, theta) [m, m, rad].

        Return:
            ds: List of magnitudes [m].
            phi: List of angles [rad].

        """
        ds = [math.sqrt(sensor[0] ** 2 + sensor[1] ** 2) for sensor in sensors]
        phi = [math.atan2(sensor[1], sensor[0]) for sensor in sensors]

        return ds, phi

    def _sense(self, particle: Tuple[float, float, float]) -> List[float]:
        """Obtains the predicted measurement of every sensor given the robot's pose.

        Args:
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns: List of predicted measurements; inf if a sensor is out of range.

        """
        rays: List[List[Tuple[float, float]]] = self._sensor_rays(particle)
        z_hat: List[float] = []

        # TODO: 2.6. Complete the missing function body with your code.
        for ray in rays:
            _, distance = self._map.check_collision(ray, compute_distance=True)
            if distance <= self._sensor_range:
                z_hat.append(distance)
            else:
                z_hat.append(float("inf"))

        return z_hat

    @staticmethod
    def _gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian value.

        """
        # TODO: 2.7. Complete the function body (i.e., replace the code below).
        return 1.0 / (
            sigma * math.sqrt(2 * math.pi) * math.e ** (((x - mu) ** 2) / (2 * (sigma**2)))
        )

    def _measurement_probability(
        self, measurements: List[float], particle: Tuple[float, float, float]
    ) -> float:
        """Computes the probability of a set of measurements given a particle's pose.

        If a measurement is unavailable (usually because it is out of range), it is replaced with
        1.25 times the sensor range to perform the computation. This value has experimentally been
        proven valid to deal with missing measurements. Nevertheless, it might not be the optimal
        replacement value.

        Args:
            measurements: Sensor measurements [m].
            particle: Particle pose (x, y, theta) [m, m, rad].

        Returns:
            float: Probability.

        """
        probability = 1.0

        # TODO: 2.8. Complete the missing function body with your code.
        for act_measure, pred_measure in zip(measurements, self._sense(particle)):
            if act_measure >= self._sensor_range:
                act_measure = 1.25 * self._sensor_range
            if pred_measure >= self._sensor_range:
                pred_measure = 1.25 * self._sensor_range
            probability = probability * self._gaussian(
                mu=pred_measure, sigma=self._sigma_z, x=act_measure
            )
        return probability

    def _sensor_rays(self, particle: Tuple[float, float, float]) -> List[List[Tuple[float, float]]]:
        """Determines the simulated sensor ray segments for a given particle.

        Args:
            particle: Particle pose (x, y, theta) in [m] and [rad].

        Returns: Ray segments. Format:
                 [[(x0_begin, y0_begin), (x0_end, y0_end)],
                  [(x1_begin, y1_begin), (x1_end, y1_end)],
                  ...]

        """
        x = particle[0]
        y = particle[1]
        theta = particle[2]

        # Convert sensors to world coordinates
        xw = [x + ds * math.cos(theta + phi) for ds, phi in zip(self._ds, self._phi)]
        yw = [y + ds * math.sin(theta + phi) for ds, phi in zip(self._ds, self._phi)]
        tw = [sensor[2] for sensor in self._sensors]

        rays = []

        for xs, ys, ts in zip(xw, yw, tw):
            x_end = xs + self._sensor_range * math.cos(theta + ts)
            y_end = ys + self._sensor_range * math.sin(theta + ts)
            rays.append([(xs, ys), (x_end, y_end)])

        return rays
