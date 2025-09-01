
import numpy as np
from scipy.spatial import distance_matrix
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time

np.random.seed(int(time.time()))

def init_weights(box_size, grid_res):
    sigma, mu = 1.0, 0
    W = (np.random.randn(grid_res, grid_res) * sigma) + mu
    x = np.linspace(0, box_size, grid_res)
    y = x
    return RegularGridInterpolator((x, y), W)

n_particles = 500
box_size = 100
grid_res = 128
interaction_radius = 5
speed = 1.0
dt = 1.0
noise = 0.001
positions = np.random.rand(n_particles, 2) * box_size
angles = np.random.rand(n_particles) * 2 * np.pi
W_fn = init_weights(box_size, grid_res)

def convert_to_velocities(angles):
    v = (np.array((np.cos(angles), np.sin(angles))) * speed).T
    return v

def convert_to_angles(velocities):
    return np.arctan2(velocities[:, 1], velocities[:, 0])

def update_by_velocities(angles, D, noise=0.01):
    # Compute velocities on the fly, use angles to keep it as unit vector
    velocities = convert_to_velocities(angles)
    velocities = (D @ velocities) / D.sum(axis=1, keepdims=True)
    noise_terms = np.random.rand(*velocities.shape) * noise
    velocities += noise_terms
    angles = convert_to_angles(velocities)
    return angles, velocities

def update_by_angles(angles, D, noise=0.01):
    # Directly get average angle
    # TODO: Does not work well.  Why?
    angles = (D @ angles) / D.sum(axis=1)
    noise_terms = (np.random.rand(n_particles) * 2 * np.pi - np.pi) * noise
    angles += noise_terms
    velocities = convert_to_velocities(angles)
    return angles, velocities

def get_weighted_coeff(positions, W_fn, min_dist=1e-9, alpha=1.0, beta=2.0):
    D = distance_matrix(positions, positions, p=2)
    zero_indices = (D < min_dist)
    D[zero_indices] = 1.
    w = W_fn(positions).reshape(-1, 1)
    diff_W = w - w.T
    D = diff_W * alpha / np.exp(D)
    D = np.abs(D)
    #print(D.min(), D.max(), D.mean())
    D[zero_indices] = 1.
    np.fill_diagonal(D, beta) # Strong diagonal value causes the particles to 'resist' change
    return D

def get_classic_coeff(positions, interaction_radius):
    D = distance_matrix(positions, positions, p=2)
    neighbor_indices = (D < interaction_radius)
    D[...] = 0
    D[neighbor_indices] = 1
    return D

def update(frame):
    global positions, angles
    #D = get_classic_coeff(positions, interaction_radius)
    D = get_weighted_coeff(positions, W_fn)
    angles, velocities = update_by_velocities(angles, D, noise=noise)
    #angles, velocities = update_by_angles(angles, D, noise=noise)
    positions += (velocities * dt)
    positions %= box_size # periodic
    quiver.set_offsets(positions)
    quiver.set_UVC(velocities[:, 0], velocities[:, 1], angles)

    return quiver

fig, ax = plt.subplots()
v = convert_to_velocities(angles)
quiver = ax.quiver(positions[:, 0], positions[:, 1], v[:, 0], v[:, 1], angles)
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)

ani = FuncAnimation(fig=fig, func=update, frames=40, interval=30)
plt.show()
