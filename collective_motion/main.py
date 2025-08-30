import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Slider
from vicsek import VicsekModel

# --- Model Parameters ---
BOX_SIZE = 100
REPULSION_RADIUS = 0.75 # This is not used in the current VicsekModel class, but was in original
INTERACTION_RADIUS = 10.0
P_DENSITY = 0.15
DT = 1.0
NOISE = 0.05 # Low noise for ordered state

N_PARTICLES = int(P_DENSITY * (BOX_SIZE ** 2))

# --- Initialize Vicsek Model ---
vicsek_model = VicsekModel(
    n_particles=N_PARTICLES,
    box_size=BOX_SIZE,
    interaction_radius=INTERACTION_RADIUS,
    noise=NOISE,
    dt=DT,
    boundary_mode="Reflective"
)

# --- Set up the Plot with Two Subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Subplot 1: Particles
title = ax1.set_title(f"Vicsek Model {vicsek_model.boundary_mode} Boundaries")
quiver = ax1.quiver(vicsek_model.positions[:, 0], vicsek_model.positions[:, 1],
                    vicsek_model.velocities[:, 0], vicsek_model.velocities[:, 1],
                    vicsek_model.speed, scale=50, cmap='coolwarm')
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
                          vmin=0, vmax=np.max(initial_density) * 1.2) # Set initial color limits
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_aspect('equal')
fig.colorbar(density_plot, ax=ax2, label="Density")

# --- Radio Buttons for Boundary Control ---
ax_radio = plt.axes([0.2, 0.03, 0.2, 0.08]) # [left, bottom, width, height]
radio_buttons = RadioButtons(ax_radio, ('Reflective', 'Periodic'))

def on_boundary_change(label):
    vicsek_model.boundary_mode = label
    ax1.set_title(f"Vicsek Model {label} Boundaries")
    fig.canvas.draw_idle()

radio_buttons.on_clicked(on_boundary_change)

# --- Slider for simulation time Control ---
ax_slider = plt.axes([0.5, 0.05, 0.2, 0.02])
slider = Slider(ax_slider, label='DT', valmin=0., valmax=2.0, valinit=vicsek_model.dt)

def on_dt_change(val):
    vicsek_model.dt = slider.val

slider.on_changed(on_dt_change)

# --- Animation Update Function ---
def update(frame):
    positions, angles, velocities, speed = vicsek_model.step()

    # Update particle plot
    quiver.set_offsets(positions)
    quiver.set_UVC(velocities[:, 0], velocities[:, 1], speed * 10)

    # Update density plot
    density_map = vicsek_model.get_density_map()
    density_plot.set_data(density_map.T)
    density_plot.set_clim(vmin=0, vmax=np.max(density_map) * 1.2)

    # Update labels
    noise_label.set_text(f'noise: {vicsek_model.noise:.2f}')
    repulse_label.set_text(f'repulse: {REPULSION_RADIUS}') # REPULSION_RADIUS is not used in model, but kept for consistency

    return quiver, density_plot, noise_label, repulse_label

# --- Run Animation ---
animation = FuncAnimation(fig, update, frames=200, interval=30, blit=True)
plt.show()
