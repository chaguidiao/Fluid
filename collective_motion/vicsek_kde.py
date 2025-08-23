import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

# --- Model Parameters ---
N_PARTICLES = 350          # Number of particles
BOX_SIZE = 100.0           # Size of the simulation box
INTERACTION_RADIUS = 10.0  # Radius for local alignment
SPEED = 1.0                # Constant speed of particles
DT = 1.0                   # Time step
NOISE = 0.3                # Noise level (low for ordered state)

# --- Density Plot Parameters ---
GRID_RESOLUTION = 100      # Resolution of the density grid (e.g., 100x100)
KDE_BANDWIDTH = 0.25       # Bandwidth for the KDE. Controls smoothness.
                           # Smaller values = sharper peaks; larger = smoother.

# --- Initialization ---
# Initialize particle positions randomly
positions = np.random.rand(N_PARTICLES, 2) * BOX_SIZE

# Initialize particle angles (and thus velocities) randomly
angles = np.random.rand(N_PARTICLES) * 2 * np.pi
velocities = np.array([np.cos(angles), np.sin(angles)]).T * SPEED

# --- Set up the Evaluation Grid for KDE (Done Once for Efficiency) ---
# We create the grid of points where we will calculate the density.
x_grid, y_grid = np.mgrid[0:BOX_SIZE:GRID_RESOLUTION*1j, 0:BOX_SIZE:GRID_RESOLUTION*1j]
grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])

# --- Set up the Plot with Two Subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Subplot 1: Particle quiver plot
ax1.set_title("Vicsek Model Particles")
quiver = ax1.quiver(positions[:, 0], positions[:, 1],
                    velocities[:, 0], velocities[:, 1],
                    angles, scale=50, cmap='hsv')
ax1.set_xlim(0, BOX_SIZE)
ax1.set_ylim(0, BOX_SIZE)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_aspect('equal')

# Subplot 2: Density field plot
ax2.set_title("KDE Density Field")
# Start with an empty plot. The 'imshow' object will be updated each frame.
initial_density = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))
density_plot = ax2.imshow(initial_density.T, origin='lower',
                          extent=[0, BOX_SIZE, 0, BOX_SIZE], cmap='viridis',
                          vmin=0, vmax=0.01) # Set initial color limits
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_aspect('equal')
fig.colorbar(density_plot, ax=ax2, label="Density")
plt.tight_layout()

# --- Animation Update Function ---
def update(frame):
    """Performs one step of the simulation and updates the plots."""
    global positions, velocities, angles

    # --- 1. Vicsek Model Update Logic ---
    # Find neighbors using a k-d tree for efficiency with periodic boundaries
    tree = cKDTree(positions, boxsize=[BOX_SIZE, BOX_SIZE])
    neighbor_indices = tree.query_ball_point(positions, r=INTERACTION_RADIUS)
    
    mean_angles = angles.copy()
    for i in range(N_PARTICLES):
        neighbors = neighbor_indices[i]
        if len(neighbors) > 0:
            # Calculate the average direction vector of the neighbors
            avg_vector = velocities[neighbors].mean(axis=0)
            # The new angle is the angle of this average vector
            mean_angles[i] = np.arctan2(avg_vector[1], avg_vector[0])

    # Add noise to the new angles
    noise_term = (np.random.rand(N_PARTICLES) - 0.5) * NOISE
    angles = mean_angles + noise_term
    
    # Update velocities and positions
    velocities = np.array([np.cos(angles), np.sin(angles)]).T * SPEED
    positions += velocities * DT
    
    # Enforce periodic boundary conditions
    positions %= BOX_SIZE

    # --- 2. Density Field Calculation using KDE ---
    # Create the KDE object from the current particle positions.
    # The data must be transposed to shape (n_dims, n_points).
    kde = gaussian_kde(positions.T, bw_method=KDE_BANDWIDTH)

    # Evaluate the KDE on our pre-defined grid.
    density_map = kde(grid_points).reshape(x_grid.shape)

    # --- 3. Update Plots ---
    # Update particle plot
    quiver.set_offsets(positions)
    quiver.set_UVC(velocities[:, 0], velocities[:, 1], np.arctan2(velocities[:, 1], velocities[:, 0]))

    # Update density plot
    density_plot.set_data(density_map.T)
    # Dynamically adjust color limits to the data range for better visualization
    density_plot.set_clim(vmin=0, vmax=np.max(density_map))

    return quiver, density_plot

# --- Run the Animation ---
# Note: blit=True can sometimes cause issues with imshow on some backends.
# If you experience problems, try setting blit=False.
animation = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()
