import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
#from scip.stats import skewnorm

class VicsekModel:
    def __init__(self, n_particles, box_size, interaction_radius, noise, dt, min_speed=-0.5, max_speed=2.5,
                 boundary_mode="Reflective", noise_decay_rate=1e-3, bounce_force=0.01, time_lag=0.01):
        self.n_particles = n_particles
        self.box_size = box_size
        self.interaction_radius = interaction_radius
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed = np.random.rand(self.n_particles) + self.min_speed
        self.noise = noise
        self.dt = dt
        self.boundary_mode = boundary_mode
        self.noise_decay_rate = noise_decay_rate
        self.bounce_force = bounce_force
        self.time_lag = time_lag
        # For density field
        self.grid_resolution = 128
        self.smoothing_sigma = 3.0

        # Initialization
        self.positions = np.random.rand(self.n_particles, 2) * self.box_size
        self.angles = np.random.rand(self.n_particles) * 2 * np.pi
        self.velocities = (np.array([np.cos(self.angles), np.sin(self.angles)]) * self.speed).T
        self.neighbor_indices = []

        # Weight initialization
        excite_ratio = 0.5
        self.weights = self._generate_weights(self.grid_resolution * self.grid_resolution, excite_ratio)
        self.weights = self.weights.reshape(self.grid_resolution, -1)
        x = np.linspace(0, self.box_size, self.grid_resolution)
        y = np.linspace(0, self.box_size, self.grid_resolution)
        self.weights_fn = RegularGridInterpolator((x, y), self.weights)

    @staticmethod
    def _generate_weights(n_samples: int, excite_ratio: float):
        assert (excite_ratio > 0) and (excite_ratio < 1.)
        samples = np.zeros(n_samples)
        positive_weights = np.random.rand(int(n_samples * excite_ratio))
        negative_weights = -np.random.rand(int(n_samples * (1 - excite_ratio)))
        remaining_n_samples = n_samples - positive_weights.size - negative_weights.size
        remaining_weights = np.random.rand(remaining_n_samples)
        weights = np.concatenate((positive_weights, negative_weights, remaining_weights))
        np.random.shuffle(weights)
        return weights

    def _get_neighbor_indices(self):
        tree = cKDTree(self.positions, boxsize=[self.box_size, self.box_size])
        return tree.query_ball_point(self.positions, r=self.interaction_radius)

    def _calculate_mean_angles(self):
        mean_angles = np.zeros_like(self.angles)
        for i in range(self.n_particles):
            # Average velocities based on the neighbors
            avg_vector = np.mean(self.velocities[self.neighbor_indices[i]], axis=0)
            mean_angles[i] = np.arctan2(avg_vector[1], avg_vector[0])
        return mean_angles

    def _calculate_weighted_speedgain(self):
        weighted_speedgain = np.zeros_like(self.speed)
        for i in range(self.n_particles):
            w = self.weights_fn(self.positions[self.neighbor_indices[i]])
            weighted_speedgain[i] = np.mean(w * self.speed[self.neighbor_indices[i]], axis=0)
        return weighted_speedgain * 0.3

    def _apply_boundary_conditions(self):
        if self.boundary_mode == "Periodic":
            pass
        else: # Reflective
            # X-axis boundaries
            left_bound_indices = self.positions[:, 0] < 0
            self.positions[left_bound_indices, 0] = self.bounce_force
            self.angles[left_bound_indices] = np.pi - self.angles[left_bound_indices]

            right_bound_indices = self.positions[:, 0] > self.box_size
            self.positions[right_bound_indices, 0] = self.box_size - self.bounce_force
            self.angles[right_bound_indices] = np.pi - self.angles[right_bound_indices]

            # Y-axis boundaries
            bottom_bound_indices = self.positions[:, 1] < 0
            self.positions[bottom_bound_indices, 1] = self.bounce_force
            self.angles[bottom_bound_indices] = -self.angles[bottom_bound_indices]

            top_bound_indices = self.positions[:, 1] > self.box_size
            self.positions[top_bound_indices, 1] = self.box_size - self.bounce_force
            self.angles[top_bound_indices] = -self.angles[top_bound_indices]

            # Update velocities again
            self.velocities = np.array([np.cos(self.angles), np.sin(self.angles)]).T

        # Ensure angles are within [0, 2*pi]
        # and positions are within box
        self.angles = self.angles % (2 * np.pi)
        self.positions %= self.box_size

    def step(self):
        self.neighbor_indices = self._get_neighbor_indices()

        # Get mean angles
        mean_angles = self._calculate_mean_angles()

        # Add decaying noise
        self.noise = np.maximum((self.noise - self.noise_decay_rate), 0)
        noise_term = (np.random.rand(self.n_particles) - 0.5) * self.noise

        # Update angles with time lag
        target_angles = (mean_angles + noise_term)
        offset_angles = (target_angles - self.angles) % (2 * np.pi)
        self.angles += (offset_angles * (1.0 - (self.time_lag % 1.0)))

        # Update speed
        self.speed += self._calculate_weighted_speedgain()
        self.speed = np.clip(self.speed, self.min_speed, self.max_speed)

        # Update velocities and positions
        self.velocities = (np.array([np.cos(self.angles), np.sin(self.angles)]) * self.speed).T
        self.positions += self.velocities * self.dt

        # Apply boundary conditions
        self._apply_boundary_conditions()

        return self.positions, self.angles, self.velocities, self.speed

    def get_density_map(self):
        # This method is kept for completeness but is decoupled from the core step
        hist, _, _ = np.histogram2d(
            self.positions[:, 0], self.positions[:, 1],
            bins=self.grid_resolution, range=[[0, self.box_size], [0, self.box_size]]
        )
        density_map = gaussian_filter(hist, sigma=self.smoothing_sigma)
        return density_map
