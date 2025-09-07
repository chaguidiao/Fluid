import taichi as ti
import numpy as np
# from scipy.spatial import cKDTree # Not used in exp_ti.py core
from scipy.ndimage import gaussian_filter # Used for density map
# from scipy.interpolate import RegularGridInterpolator # Not used in exp_ti.py core
import opensimplex # Used for W initialization
from perlin_numpy import (
    generate_perlin_noise_2d,
    generate_fractal_noise_2d
)

# Initialize Taichi once globally
ti.init(arch=ti.vulkan)

@ti.data_oriented # Add this decorator
class TaichiVicsekModel:
    def __init__(self, n_particles, n_forces, box_size, dt, max_speed, min_speed, weight_offset, grid_res=128, alpha=1.0, accelerate_factor=None):
        self.n_particles = n_particles
        self.n_forces = n_forces
        self.box_size = box_size
        self.dt = dt
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.weight_offset = weight_offset
        self.grid_res = grid_res
        self.alpha = alpha
        if not accelerate_factor:
            self.accelerate_factor = 0.001 * self.box_size
        else:
            self.accelerate_factor = accelerate_factor
        self.t = 0

        # Taichi fields (equivalent to global fields in exp_ti.py)
        self.positions = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles)
        self.force_positions = ti.Vector.field(2, dtype=ti.f32, shape=self.n_forces)
        self.angles = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.speed = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.W = ti.field(dtype=ti.f32, shape=(self.grid_res, self.grid_res))
        self.row_sums = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.wpos_field = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.velocities_field = ti.Vector.field(2, dtype=ti.f32, shape=self.n_particles)
        self.force_field = ti.Vector.field(2, dtype=ti.f32, shape=n_forces)
        self.C_field = ti.field(dtype=ti.f32, shape=(self.n_particles, self.n_particles))
        self.F_field = ti.field(dtype=ti.f32, shape=(self.n_particles, self.n_forces))

        # Initialize fields on CPU and copy to Taichi
        W_np = generate_fractal_noise_2d(
            (self.grid_res, self.grid_res),
            (8, 8),
            5
        )
        W_np += self.weight_offset
        self.W.from_numpy(W_np.astype(np.float32))

        positions_np = np.random.rand(self.n_particles, 2) * self.box_size
        angles_np = np.random.rand(self.n_particles) * 2 * np.pi
        speed_np = np.repeat(self.max_speed, self.n_particles)

        self.positions.from_numpy(positions_np.astype(np.float32))
        self.angles.from_numpy(angles_np.astype(np.float32))
        self.speed.from_numpy(speed_np.astype(np.float32))

        # Store initial numpy arrays for plotting setup in main.py
        self.initial_positions_np = positions_np
        self.initial_angles_np = angles_np
        self.initial_velocities_np = (np.array([np.cos(angles_np), np.sin(angles_np)])).T
        self.initial_W_np = W_np # For heatmap

    def set_forces(self, force_positions: np.ndarray, forces: np.ndarray):
        self.force_positions.from_numpy(force_positions.astype(np.float32))
        self.force_field.from_numpy(forces.astype(np.float32))

    @ti.func
    def get_weight_at_pos(self, pos):
        x_norm = pos[0] / self.box_size * (self.grid_res - 1)
        y_norm = pos[1] / self.box_size * (self.grid_res - 1)

        x0, y0 = int(x_norm), int(y_norm)
        x1, y1 = x0 + 1, y0 + 1

        x0 = max(0, min(self.grid_res - 1, x0))
        y0 = max(0, min(self.grid_res - 1, y0))
        x1 = max(0, min(self.grid_res - 1, x1))
        y1 = max(0, min(self.grid_res - 1, y1))

        q00 = self.W[x0, y0]
        q10 = self.W[x1, y0]
        q01 = self.W[x0, y1]
        q11 = self.W[x1, y1]

        tx = x_norm - x0
        ty = y_norm - y0

        return (1 - tx) * (1 - ty) * q00 + tx * (1 - ty) * q10 + \
               (1 - tx) * ty * q01 + tx * ty * q11

    @ti.func
    def get_wpos(self):
        for i in range(self.n_particles):
            self.wpos_field[i] = self.get_weight_at_pos(self.positions[i])

    @ti.func
    def convert_to_velocities(self):
        for i in range(self.n_particles):
            self.velocities_field[i][0] = ti.cos(self.angles[i])
            self.velocities_field[i][1] = ti.sin(self.angles[i])

    @ti.func
    def convert_to_angles(self):
        for i in range(self.n_particles):
            self.angles[i] = ti.atan2(self.velocities_field[i][1],
                                      self.velocities_field[i][0])

    @ti.func
    def get_weighted_coeff(self):
        for i, j in ti.ndrange(self.n_particles, self.n_particles):
            r = self.positions[i] - self.positions[j]
            dist_sq = r[0]**2 + r[1]**2
            D_val = 1.0 / ti.exp(ti.sqrt(dist_sq))
            self.C_field[i, j] = D_val

        for i in range(self.n_particles):
            row_sum = 0.0
            for j in range(self.n_particles):
                row_sum += self.C_field[i, j]
            self.row_sums[i] = row_sum

        for i, j in ti.ndrange(self.n_particles, self.n_particles):
            if self.row_sums[i] > 0:
                self.C_field[i, j] /= self.row_sums[i]
            else:
                self.C_field[i, j] = 0.0

        wpos_norm_l2_sq = 0.0
        for i in range(self.n_particles):
            wpos_norm_l2_sq += self.wpos_field[i]**2
        wpos_norm_l2 = ti.sqrt(wpos_norm_l2_sq)
        if wpos_norm_l2 == 0:
            wpos_norm_l2 = 1.0

        for i, j in ti.ndrange(self.n_particles, self.n_particles):
            # alpha controls how 'fast' the info spreads throughout the field
            wposnorm_val = self.wpos_field[j] / wpos_norm_l2
            self.C_field[i, j] += self.alpha * wposnorm_val**3

    @ti.func
    def update_alignments(self):
        for i in range(self.n_particles):
            new_vel_x = 0.0
            new_vel_y = 0.0
            for j in range(self.n_particles):
                new_vel_x += self.C_field[i, j] * self.velocities_field[j][0]
                new_vel_y += self.C_field[i, j] * self.velocities_field[j][1]
            self.velocities_field[i][0] = new_vel_x
            self.velocities_field[i][1] = new_vel_y

    @ti.func
    def get_acceleration(self):
        for i in range(self.n_particles):
            ds = 0.0
            for j in range(self.n_particles):
                ds += self.C_field[i, j] * self.wpos_field[j]
            self.speed[i] += ds * self.accelerate_factor

    @ti.func
    def update_positions(self):
        for i in range(self.n_particles):
            self.velocities_field[i][0] *= self.speed[i]
            self.velocities_field[i][1] *= self.speed[i]

            self.positions[i][0] += self.velocities_field[i][0] * self.dt
            self.positions[i][1] += self.velocities_field[i][1] * self.dt

            self.positions[i][0] %= self.box_size
            self.positions[i][1] %= self.box_size

    @ti.func
    def clip_speed(self):
        for i in range(self.n_particles):
            self.speed[i] = ti.max(self.min_speed, ti.min(self.max_speed, self.speed[i]))

    @ti.func
    def apply_forces(self):
        for i, j in ti.ndrange(self.n_particles, self.n_forces):
            r = self.positions[i] - self.force_positions[j]
            dist_sq = r[0]**2 + r[1]**2
            D_val = 1.0 / ti.exp(ti.sqrt(dist_sq))
            self.F_field[i, j] = D_val

        t_decay = ti.max(0, 5. - 0.1 * self.t)
        for i in ti.ndrange(self.n_particles):
            for j in range(self.n_forces):
                dx = self.F_field[i, j] * self.force_field[j][0] * t_decay
                dy = self.F_field[i, j] * self.force_field[j][1] * t_decay
                self.velocities_field[i][0] += dx
                self.velocities_field[i][1] += dx
                self.speed[i] = ti.sqrt(self.velocities_field[i][0]**2 + self.velocities_field[i][1]**2)


    @ti.kernel
    def update(self):
        self.get_wpos()
        self.get_weighted_coeff()
        self.convert_to_velocities()
        self.update_alignments()
        self.get_acceleration()
        self.clip_speed()
        self.update_positions()
        self.apply_forces()
        self.convert_to_angles()

    def step(self):
        self.t += 1
        self.update()

        # Return numpy arrays for plotting
        return self.positions.to_numpy(), self.angles.to_numpy(), \
               self.velocities_field.to_numpy(), self.speed.to_numpy()

    def get_density_map(self):
        # This method is kept for completeness but is decoupled from the core step
        # and still uses numpy/scipy for now.
        # Need to convert positions to numpy for histogram2d
        positions_np = self.positions.to_numpy()
        hist, _, _ = np.histogram2d(
            positions_np[:, 0], positions_np[:, 1],
            bins=self.grid_res, range=[[0, self.box_size], [0, self.box_size]]
        )
        # Assuming gaussian_filter is still desired for density map
        density_map = gaussian_filter(hist, sigma=3.0) # Use a default sigma for now
        return density_map

    def get_weights_np(self):
        # Return the W field as a numpy array for plotting
        return self.W.to_numpy()
