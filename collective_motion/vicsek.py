import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Slider
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# --- Model Parameters ---
BOX_SIZE = 100
REPULSION_RADIUS = 0.75
REPULSION_RADIUS_MIN = 0.55
INTERACTION_RADIUS = 10.0
time_lag = 0.01
P_DENSITY = 0.15
SPEED = 1.0
DT = 1.0
BOUNCE_FORCE = 15.0 # Some small bounce force to prevent particles to stick alongside the wall?
NOISE = 0.35 # Low noise for ordered state
NOISE_DECAY_RATE = 1e-3
current_boundary_mode = "Reflective"
N_PARTICLES = int(P_DENSITY * (BOX_SIZE ** 2))

# --- Density Plot Parameters ---
GRID_RESOLUTION = 128
SMOOTHING_SIGMA = 3.0

# --- Initialization ---
positions = np.random.rand(N_PARTICLES, 2) * BOX_SIZE
angles = np.random.rand(N_PARTICLES) * 2 * np.pi
velocities = np.array([np.cos(angles), np.sin(angles)]).T * SPEED
#weights = np.random.rand(GRID_RESOLUTION, GRID_RESOLUTION) * 2 - 1.0 # Snap to -1.0 to 1.0
weights = np.random.rand(GRID_RESOLUTION, GRID_RESOLUTION)
x = np.linspace(0, BOX_SIZE, GRID_RESOLUTION)
y = np.linspace(0, BOX_SIZE, GRID_RESOLUTION)
weights_fn = RegularGridInterpolator((x, y), weights)

# --- Set up the Plot with Two Subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Subplot 1: Particles
title = ax1.set_title(f"Vicsek Model {current_boundary_mode} Boundaries")
quiver = ax1.quiver(positions[:, 0], positions[:, 1],
                    velocities[:, 0], velocities[:, 1],
                    angles, scale=50, cmap='hsv')
ax1.set_xlim(0, BOX_SIZE)
ax1.set_ylim(0, BOX_SIZE)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_aspect('equal')

noise_label = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, ha='left', va='top')
repulse_label = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, ha='left', va='top')

# Subplot 2: Density Field
ax2.set_title("Smoothed Density Field")
# Create an empty initial density map
initial_density = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))
density_plot = ax2.imshow(initial_density.T, origin='lower',
                          extent=[0, BOX_SIZE, 0, BOX_SIZE], cmap='viridis',
                          vmin=0, vmax=0.1) # Set initial color limits
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_aspect('equal')
fig.colorbar(density_plot, ax=ax2, label="Density")
#plt.tight_layout()

# --- Radio Buttons for Boundary Control ---
ax_radio = plt.axes([0.2, 0.03, 0.2, 0.08]) # [left, bottom, width, height]
radio_buttons = RadioButtons(ax_radio, ('Reflective', 'Periodic'))

def on_boundary_change(label):
    """Callback function to change boundary condition."""
    global current_boundary_mode
    current_boundary_mode = label
    ax1.set_title(f"Vicsek Model {label} Boundaries")
    fig.canvas.draw_idle()

radio_buttons.on_clicked(on_boundary_change)

# --- Slider for simulation time Control ---
ax_slider = plt.axes([0.5, 0.05, 0.2, 0.02])
slider = Slider(ax_slider, label='DT', valmin=0., valmax=2.0, valinit=1.)

def on_dt_change(val):
    global DT
    DT = slider.val

slider.on_changed(on_dt_change)

# --- Animation Update Function ---
def update(frame):
    global positions, velocities, angles, title, NOISE, REPULSION_RADIUS

    # --- Vicsek Model Update (same as before) ---
    assert np.all(positions > 0)
    assert np.all(positions < BOX_SIZE)
    tree = cKDTree(positions, boxsize=[BOX_SIZE, BOX_SIZE])
    mean_angles = angles.copy()
    neighbor_indices = tree.query_ball_point(positions, r=INTERACTION_RADIUS)
    REPULSION_RADIUS = np.maximum((REPULSION_RADIUS - 0.01), REPULSION_RADIUS_MIN)
    repulse_neighbor_indices = tree.query_ball_point(positions, r=REPULSION_RADIUS)
    for i in range(N_PARTICLES):
        '''
        # Repulsion takes precedence over alignment
        if len(repulse_neighbor_indices[i]) > 1:
            avg_vector = (positions[i] - positions[repulse_neighbor_indices[i]]).mean(axis=0)
            mean_angles[i] = np.arctan2(avg_vector[1], avg_vector[0])
        elif len(neighbor_indices[i]) > 1:
            avg_vector = velocities[neighbor_indices[i]].mean(axis=0)
            mean_angles[i] = np.arctan2(avg_vector[1], avg_vector[0])
        '''
        avg_vector_repulse = np.zeros_like(velocities[i])
        avg_vector_align = np.zeros_like(velocities[i])
        if len(repulse_neighbor_indices[i]) > 1:
            avg_vector_repulse = (positions[i] - positions[repulse_neighbor_indices[i]]).mean(axis=0)
        if len(neighbor_indices[i]) > 1:
            avg_vector_align = velocities[neighbor_indices[i]].mean(axis=0)

        w = weights_fn(positions[i])
        avg_vector = (1. - w) * avg_vector_repulse + w * avg_vector_align
        mean_angles[i] = np.arctan2(avg_vector[1], avg_vector[0])

    # Add slow decaying noise
    NOISE = np.maximum((NOISE - NOISE_DECAY_RATE), 0)
    noise_term = (np.random.rand(N_PARTICLES) - 0.5) * NOISE

    #angles = mean_angles + noise_term
    target_angles = (mean_angles + noise_term)
    offset_angles = (target_angles - angles) % (2 * np.pi)
    angles += (offset_angles * (1.0 - (time_lag % 1.0)))
    velocities = np.array([np.cos(angles), np.sin(angles)]).T * SPEED
    positions += velocities * DT

    if current_boundary_mode == "Periodic":
        # --- Periodic boundary conditions ---
        pass
    else:
        # --- Bouncing boundary conditions ---
        for i in range(N_PARTICLES):
            # Check for x-axis boundaries
            if positions[i, 0] < 0:
                positions[i, 0] = -positions[i, 0] * BOUNCE_FORCE
                angles[i] = np.pi - angles[i] # Reflect angle
            elif positions[i, 0] > BOX_SIZE:
                positions[i, 0] = 2 * BOX_SIZE - positions[i, 0] * BOUNCE_FORCE
                angles[i] = np.pi - angles[i] # Reflect angle

            # Check for y-axis boundaries
            if positions[i, 1] < 0:
                positions[i, 1] = -positions[i, 1] * BOUNCE_FORCE
                angles[i] = -angles[i] # Reflect angle
            elif positions[i, 1] > BOX_SIZE:
                positions[i, 1] = 2 * BOX_SIZE - positions[i, 1] * BOUNCE_FORCE
                angles[i] = -angles[i] # Reflect angle

    positions %= BOX_SIZE
    angles = angles % (2 * np.pi) # Ensure headings are within the [0, 2*pi] range
    velocities = np.array([np.cos(angles), np.sin(angles)]).T

    # --- Density Field Calculation ---
    # 1. Bin data into a histogra
    hist, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=GRID_RESOLUTION, range=[[0, BOX_SIZE], [0, BOX_SIZE]]
    )
    # 2. Smooth the histogram
    density_map = gaussian_filter(hist, sigma=SMOOTHING_SIGMA)

    # --- Update Plots ---
    # Update particle plot
    quiver.set_offsets(positions)
    quiver.set_UVC(velocities[:, 0], velocities[:, 1], angles)

    # Update density plot
    density_plot.set_data(density_map.T)
    # Optional: dynamically adjust color limits for better visualization
    density_plot.set_clim(vmin=0, vmax=np.max(density_map) * 0.8)

    # Update labels
    noise_label.set_text(f'noise: {NOISE}')
    repulse_label.set_text(f'repulse: {REPULSION_RADIUS}')

    return quiver, density_plot, noise_label, repulse_label

# --- Run Animation ---
animation = FuncAnimation(fig, update, frames=200, interval=30, blit=False)
plt.show()
