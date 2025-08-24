import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

class VicsekModel:
    def __init__(self, n_particles, box_size, interaction_radius, speed, noise, dt,
                 boundary_mode="Reflective", noise_decay_rate=1e-3, bounce_force=15.0, time_lag=0.01):
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
        self.weights = np.random.rand(self.grid_resolution, self.grid_resolution)
        x = np.linspace(0, self.box_size, self.grid_resolution)
        y = np.linspace(0, self.box_size, self.grid_resolution)
        self.weights_fn = RegularGridInterpolator((x, y), self.weights)

    def _calculate_mean_angles(self):
        tree = cKDTree(self.positions, boxsize=[self.box_size, self.box_size])
        mean_angles = self.angles.copy()
        neighbor_indices = tree.query_ball_point(self.positions, r=self.interaction_radius)

        for i in range(self.n_particles):
            # Original logic from vicsek.py, using weights for average velocity
            w = self.weights_fn(self.positions[neighbor_indices[i]])[:, np.newaxis]
            avg_vector = np.einsum('ij,jk->k', w, self.velocities[neighbor_indices[i]])
            mean_angles[i] = np.arctan2(avg_vector[1], avg_vector[0])
        return mean_angles

    def _apply_boundary_conditions(self):
        if self.boundary_mode == "Periodic":
            self.positions %= self.box_size
        else: # Reflective
            for i in range(self.n_particles):
                # Check for x-axis boundaries
                if self.positions[i, 0] < 0:
                    self.positions[i, 0] = -self.positions[i, 0] * self.bounce_force
                    self.angles[i] = np.pi - self.angles[i] # Reflect angle
                elif self.positions[i, 0] > self.box_size:
                    self.positions[i, 0] = 2 * self.box_size - self.positions[i, 0] * self.bounce_force
                    self.angles[i] = np.pi - self.angles[i] # Reflect angle

                # Check for y-axis boundaries
                if self.positions[i, 1] < 0:
                    self.positions[i, 1] = -self.positions[i, 1] * self.bounce_force
                    self.angles[i] = -self.angles[i] # Reflect angle
                elif self.positions[i, 1] > self.box_size:
                    self.positions[i, 1] = 2 * self.box_size - self.positions[i, 1] * self.bounce_force
                    self.angles[i] = -self.angles[i] # Reflect angle
            self.positions %= self.box_size # Ensure positions are within box after bounce

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

        # Ensure angles are within [0, 2*pi]
        self.angles = self.angles % (2 * np.pi)

        return self.positions, self.angles, self.velocities

    def get_density_map(self):
        # This method is kept for completeness but is decoupled from the core step
        hist, _, _ = np.histogram2d(
            self.positions[:, 0], self.positions[:, 1],
            bins=self.grid_resolution, range=[[0, self.box_size], [0, self.box_size]]
        )
        density_map = gaussian_filter(hist, sigma=self.smoothing_sigma)
        return density_map
