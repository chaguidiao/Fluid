
import numpy as np
from scipy.spatial import distance_matrix
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter # New import
import opensimplex

np.random.seed(int(time.time()))
opensimplex.seed(int(time.time()))

def get_weight_fn(W):
    x = np.linspace(0, box_size, grid_res)
    y = x
    return RegularGridInterpolator((x, y), W)

n_particles = 500
box_size = 50
grid_res = 128
dt = 1.0
noise = 0.
max_speed = 2.5
min_speed = 0.0
weight_offset = -0.3 # Negative offset decreasing time to converge (stable down)
W = opensimplex.noise2array(
    np.linspace(0, box_size, grid_res),
    np.linspace(0, box_size, grid_res)
)
W += weight_offset
positions = np.random.rand(n_particles, 2) * box_size
angles = np.random.rand(n_particles) * 2 * np.pi
#speed = np.random.rand(n_particles) * max_speed
speed = np.abs(get_weight_fn(W)(positions)) * max_speed

def convert_to_velocities(angles):
    v = np.array((np.cos(angles), np.sin(angles))).T
    return v

def convert_to_angles(velocities):
    return np.arctan2(velocities[:, 1], velocities[:, 0])

def get_weighted_coeff(positions):
    W_fn = get_weight_fn(W)
    w = W_fn(positions)
    D = distance_matrix(positions, positions, p=2)
    D = 1 / np.exp(D)
    Dnorm = D / np.linalg.norm(D, ord=1)
    wnorm = w / np.linalg.norm(w, ord=2)
    return Dnorm, wnorm

def update_alignments(velocities, C):
    #velocities = np.abs(C) @ velocities
    velocities = C @ velocities
    return velocities

def get_noise(noise=0.01):
    # TODO: Fix
    dv = np.random.rand(n_particles, 1) * noise
    dv = dv / np.linalg.norm(dv)
    return dv.reshape(-1, 1)

def get_cohesion(positions, D):
    # Weighted average center with respect to the particles
    # Each particles has their own 'center' to be repulsed from/attraced to
    # TODO: Fix
    force = 0.001
    center_positions = (D @ positions) / np.abs(D).sum(axis=1, keepdims=True)
    dv = (positions - center_positions) * force
    return dv

def get_acceleration(speed, C, accelerate_factor=1.0):
    # Acceleration factor affects how soon the particles stabilize
    return C @ speed * accelerate_factor

def update(frame):
    global W, positions, angles, speed
    Dnorm, wnorm = get_weighted_coeff(positions)
    velocities = convert_to_velocities(angles)
    velocities = update_alignments(velocities, Dnorm + 1.00 * wnorm**3)
    #velocities += get_cohesion(positions)
    #velocities += get_noise(noise=noise)
    speed += get_acceleration(speed, Dnorm + 1.00 * wnorm**3)
    speed = np.clip(speed, min_speed, max_speed)
    #print(speed.min(), speed.max(), speed.mean(), speed.std())
    velocities *= speed.reshape(-1, 1)
    positions += (velocities * dt)
    positions %= box_size # periodic
    angles = convert_to_angles(velocities)
    quiver_particles.set_offsets(positions)
    quiver_particles.set_UVC(velocities[:, 0], velocities[:, 1], angles)

    return quiver_particles,

fig, ax = plt.subplots()

# Heatmap of W
im = ax.imshow(W, cmap='viridis', origin='lower', extent=[0, box_size, 0, box_size], alpha=0.1)

# Smoothed W and its gradient for quivers
smoothed_W = gaussian_filter(W, sigma=5) # Apply Gaussian filter
Y, X = np.mgrid[0:box_size:grid_res*1j, 0:box_size:grid_res*1j]
dW_dx, dW_dy = np.gradient(smoothed_W, box_size/grid_res) # Calculate gradient
quiver_W = ax.quiver(X, Y, dW_dx, dW_dy, color='red', alpha=0.5) # Quivers for W gradient

# Particle quivers
v = convert_to_velocities(angles)
quiver_particles = ax.quiver(positions[:, 0], positions[:, 1], v[:, 0], v[:, 1], angles)

ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)

ani = FuncAnimation(fig=fig, func=update, frames=40, interval=30, blit=False) # blit=True for performance
plt.show()
