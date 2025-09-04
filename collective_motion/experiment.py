
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

n_particles = 500
box_size = 50
grid_res = 128
interaction_radius = 5
dt = 1.0
noise = 0.
max_speed = 2.5
min_speed = 0.0
weight_offset = -0.7 # Negative offset decreasing time to converge (stable down)
W = opensimplex.noise2array(
    np.linspace(0, box_size, grid_res),
    np.linspace(0, box_size, grid_res)
)
W += weight_offset
positions = np.random.rand(n_particles, 2) * box_size
angles = np.random.rand(n_particles) * 2 * np.pi
speed = np.random.rand(n_particles) * max_speed

def get_weight_fn(W):
    x = np.linspace(0, box_size, grid_res)
    y = x
    return RegularGridInterpolator((x, y), W)

def convert_to_velocities(angles):
    v = (np.array((np.cos(angles), np.sin(angles))) * speed).T
    return v

def convert_to_angles(velocities):
    return np.arctan2(velocities[:, 1], velocities[:, 0])

def update_alignments(angles, D, w, noise=0.01):
    # Compute velocities on the fly, use angles to keep it as unit vector
    velocities = convert_to_velocities(angles)
    D_abs = np.abs(D)
    velocities = (D_abs @ velocities) / D_abs.sum(axis=1, keepdims=True)
    noise_terms = (np.random.rand(n_particles) * 2 * np.pi - np.pi)
    velocities += convert_to_velocities(noise_terms) * noise
    angles = convert_to_angles(velocities)
    return angles, velocities

def get_weighted_coeff(positions, w):
    """
    alpha: How much influence of other particles to force alignment
    beta: How much resistance the given particle to align
    """
    D = distance_matrix(positions, positions, p=2)
    D = 1 / np.exp(D)
    wnorm = w / np.linalg.norm(w)
    D = D * wnorm
    return D

def get_acceleration(speed, D, accelerate_factor=0.5):
    # Acceleration factor affects how soon the particles stabilize
    return D @ speed * accelerate_factor

'''
def get_cohesion(positions, D):
    # Weighted average center with respect to the particles
    # Each particles has their own 'center' to be repulsed from/attraced to
    force = 0.001
    # TODO Check validity
    center_positions = (D @ positions) / np.abs(D).sum(axis=1, keepdims=True)
    v = (positions - center_positions) * force
    theta = convert_to_angles(v)
    return theta, v
'''

def get_cohesion(positions, w):
    force = 0.1
    center = positions.mean(axis=0)
    v = (positions - center) * w.reshape(-1, 1) * force
    theta = convert_to_angles(v)
    return theta, v

def update(frame):
    global W, positions, angles, speed
    W_fn = get_weight_fn(W)
    w = W_fn(positions)
    D = get_weighted_coeff(positions, w)
    speed += get_acceleration(speed, D)
    speed = np.clip(speed, min_speed, max_speed)
    angles, velocities = update_alignments(angles, D, w, noise=noise)
    #da, dv = get_cohesion(positions, w)
    #angles += da
    #velocities += dv
    positions += (velocities * dt)
    positions %= box_size # periodic
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

ani = FuncAnimation(fig=fig, func=update, frames=40, interval=30, blit=True) # blit=True for performance
plt.show()
