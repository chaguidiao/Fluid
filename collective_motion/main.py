import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Slider
from scipy.ndimage import gaussian_filter # Still needed for weights heatmap smoothing
# from vicsek import VicsekModel # Original VicsekModel
from vicsek import TaichiVicsekModel # New Taichi-based model

#np.random.seed(42)

# --- Model Parameters (adapted for exp_ti.py's model) ---
BOX_SIZE = 100.0 # From exp_ti.py
N_PARTICLES = 1000 # From exp_ti.py
N_FORCES = 32
GRID_RES = 128 # From exp_ti.py
DT = 1.0 # From exp_ti.py
MAX_SPEED = 3.0 # From exp_ti.py
MIN_SPEED = 0.0 # From exp_ti.py
WEIGHT_OFFSET = -0.3 # From exp_ti.py
ALPHA = 1.0
ACCELERATE_FACTOR = 0.001 * BOX_SIZE

# --- Gaussian Filter Configuration for Weights Heatmap ---
GAUSSIAN_SIGMA = 2.0 # From exp_ti.py

# --- Initialize TaichiVicsekModel ---
vicsek_model = TaichiVicsekModel(
    n_particles=N_PARTICLES,
    n_forces=N_FORCES,
    box_size=BOX_SIZE,
    dt=DT,
    max_speed=MAX_SPEED,
    min_speed=MIN_SPEED,
    weight_offset=WEIGHT_OFFSET,
    grid_res=GRID_RES,
    alpha=ALPHA,
    accelerate_factor=ACCELERATE_FACTOR,
)
# --- Initialize forces ---
force_positions = np.random.rand(N_FORCES, 2) * BOX_SIZE
#forces = np.random.rand(N_FORCES, 2) - 0.5 * 2.0
forces = np.random.randn(N_FORCES * 2).reshape(-1, 2) * 0.5
vicsek_model.set_forces(force_positions, forces)

# --- Set up the Plot with Three Subplots ---
fig = plt.figure(figsize=(21, 7))

ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2) # Particle plot
ax2 = plt.subplot2grid((2, 3), (0, 2)) # Density plot
ax3 = plt.subplot2grid((2, 3), (1, 2)) # Weights heatmap

# Subplot 1: Particles
title = ax1.set_title(f"Taichi Vicsek Model (Periodic Boundaries)") # Boundary mode is fixed to periodic for now
quiver = ax1.quiver(vicsek_model.initial_positions_np[:, 0], vicsek_model.initial_positions_np[:, 1],
                    vicsek_model.initial_velocities_np[:, 0], vicsek_model.initial_velocities_np[:, 1])
force_quiver = ax1.quiver(
    force_positions[:, 0], force_positions[:, 1],
    forces[:, 0], forces[:, 1],
    color='red'
)
ax1.set_xlim(0, BOX_SIZE)
ax1.set_ylim(0, BOX_SIZE)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_aspect('equal')

noise_label = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, ha='left', va='top')
repulse_label = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, ha='left', va='top')

# Subplot 2: Density Field
ax2.set_title("Smoothed Density Field")
initial_density = vicsek_model.get_density_map()
density_plot = ax2.imshow(initial_density.T, origin='lower',
                          extent=[0, BOX_SIZE, 0, BOX_SIZE], cmap='viridis',
                          vmin=0, vmax=np.max(initial_density) * 1.2)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_aspect('equal')
fig.colorbar(density_plot, ax=ax2, label="Density")

# Subplot 3: Weights Heatmap
ax3.set_title(f"Weights Heatmap")
weights_plot = ax3.imshow(vicsek_model.W.to_numpy().T, origin='lower',
                          extent=[0, BOX_SIZE, 0, BOX_SIZE], cmap='coolwarm', interpolation='lanczos')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_aspect('equal')
fig.colorbar(weights_plot, ax=ax3, label="Weight")

# --- Radio Buttons for Boundary Control ---
# Removed for now, as Taichi model currently only supports periodic
# ax_radio = plt.axes([0.2, 0.03, 0.2, 0.08])
# radio_buttons = RadioButtons(ax_radio, ('Reflective', 'Periodic'))
# def on_boundary_change(label):
#     vicsek_model.boundary_mode = label
#     ax1.set_title(f"Vicsek Model {label} Boundaries")
#     fig.canvas.draw_idle()
# radio_buttons.on_clicked(on_boundary_change)
# radio_buttons.set_active(1)

# --- Slider for simulation time Control ---
ax_slider = plt.axes([0.5, 0.05, 0.2, 0.02])
slider = Slider(ax_slider, label='DT', valmin=0., valmax=2.0, valinit=vicsek_model.dt)

def on_dt_change(val):
    vicsek_model.dt = slider.val # This will update the dt in the Taichi model
    # Note: Changing dt in Taichi fields directly from Python is not straightforward.
    # For now, this will update the Python-side dt, which is passed to the kernel.

slider.on_changed(on_dt_change)

# --- Animation Update Function ---
def update(frame):
    positions, angles, velocities, speed = vicsek_model.step()

    # Update particle plot
    quiver.set_offsets(positions)
    quiver.set_UVC(velocities[:, 0], velocities[:, 1], angles) # Use actual speed for color

    # Update force quivers
    force_quiver.set_offsets(vicsek_model.force_positions.to_numpy())
    t_decay = np.fmax(0, 5. - 0.1 * vicsek_model.t)
    f = vicsek_model.force_field.to_numpy() * t_decay
    force_quiver.set_UVC(f[:, 0], f[:, 1])

    # Update density plot
    density_map = vicsek_model.get_density_map()
    density_plot.set_data(density_map.T)
    density_plot.set_clim(vmin=0, vmax=np.max(density_map) * 1.2)

    # Update weights plot (W is static in exp_ti.py, so this won't change much)
    weights_plot.set_data(vicsek_model.W.to_numpy().T)

    # Update labels (noise and repulsion are not directly from exp_ti.py model)
    noise_label.set_text(f'noise: N/A (Taichi model)')
    repulse_label.set_text(f'repulse: N/A (Taichi model)')

    return quiver, force_quiver, density_plot, weights_plot, noise_label, repulse_label

# --- Run Animation ---
animation = FuncAnimation(fig, update, frames=200, interval=30, blit=False)
plt.show()
