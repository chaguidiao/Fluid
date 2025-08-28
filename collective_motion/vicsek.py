import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

class VicsekModel:
    def __init__(self, n_particles, box_size, interaction_radius, speed, noise, dt,
                 boundary_mode="Reflective", noise_decay_rate=1e-3, bounce_force=0.01, time_lag=0.01):
        self.n_particles = n_particles
        self.box_size = box_size
        self.interaction_radius = interaction_radius
        self.speed = speed
        self.noise = noise
        self.dt = dt
        self.boundary_mode = boundary_mode
        self.noise_decay_rate = noise_decay_rate
        self.bounce_force = bounce_force
        self.time_lag = time_lag

        # Initialization
        self.positions = np.random.rand(self.n_particles, 2) * self.box_size
        self.angles = np.random.rand(self.n_particles) * 2 * np.pi
        self.velocities = np.array([np.cos(self.angles), np.sin(self.angles)]).T * self.speed

        # For density field (if needed later, though decoupled from core model)
        self.grid_resolution = 128
        self.smoothing_sigma = 3.0
        self.weights = (np.random.rand(self.grid_resolution, self.grid_resolution) * 2) - 1
        x = np.linspace(0, self.box_size, self.grid_resolution)
        y = np.linspace(0, self.box_size, self.grid_resolution)
        self.weights_fn = RegularGridInterpolator((x, y), self.weights)

    def _calculate_mean_angles(self):
        tree = cKDTree(self.positions, boxsize=[self.box_size, self.box_size])
        mean_angles = self.angles.copy()
        neighbor_indices = tree.query_ball_point(self.positions, r=self.interaction_radius)

        for i in range(self.n_particles):
            # Average velocities affected by the weights of the neighbors
            w = self.weights_fn(self.positions[neighbor_indices[i]])[:, np.newaxis]
            np.fill_diagonal(w, 0) # Excluding self-loop
            avg_vector = np.einsum('ij,jk->k', w, self.velocities[neighbor_indices[i]])
            mean_angles[i] = np.arctan2(avg_vector[1], avg_vector[0])
        return mean_angles

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
        mean_angles = self._calculate_mean_angles()

        # Add decaying noise
        self.noise = np.maximum((self.noise - self.noise_decay_rate), 0)
        noise_term = (np.random.rand(self.n_particles) - 0.5) * self.noise

        # Update angles with time lag
        target_angles = (mean_angles + noise_term)
        offset_angles = (target_angles - self.angles) % (2 * np.pi)
        self.angles += (offset_angles * (1.0 - (self.time_lag % 1.0)))

        # Update velocities and positions
        self.velocities = np.array([np.cos(self.angles), np.sin(self.angles)]).T * self.speed
        self.positions += self.velocities * self.dt

        # Apply boundary conditions
        self._apply_boundary_conditions()

        return self.positions, self.angles, self.velocities

    def get_density_map(self):
        # This method is kept for completeness but is decoupled from the core step
        hist, _, _ = np.histogram2d(
            self.positions[:, 0], self.positions[:, 1],
            bins=self.grid_resolution, range=[[0, self.box_size], [0, self.box_size]]
        )
        density_map = gaussian_filter(hist, sigma=self.smoothing_sigma)
        return density_map
