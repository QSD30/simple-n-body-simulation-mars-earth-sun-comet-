import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Step 1: Setup and Initial Conditions ---

# Gravitational constant
G = 6.67430e-11  # m^3 kg^-1 s^-2

# Simulation parameters
N = 4  # Number of bodies
total_time = 2 * 365 * 24 * 3600  # 2 years in seconds
dt = 24 * 3600  # 1 day in seconds
num_steps = int(total_time / dt)

# Initial conditions for 4 bodies
# masses are in kg, positions in meters, velocities in m/s
masses = np.array([
    1.989e30,  # Sun
    5.972e24,  # Earth
    6.417e23,  # Mars
    1.0e20     # Comet
])

# Initial positions (x, y)
positions = np.array([
    [0, 0],
    [1.496e11, 0],
    [-2.279e11, 0],
    [3.0e11, 1.0e11]
])

# Initial velocities (vx, vy)
velocities = np.array([
    [0, 0],
    [0, 29783],
    [0, -24077],
    [-20000, -5000]
], dtype=np.float64)

# Store history for plotting
positions_history = np.zeros((num_steps, N, 2))
energy_history = np.zeros(num_steps)


# --- Step 2: Physics Engine ---

def calculate_forces(masses, positions):
    """Calculates the net gravitational force on each body."""
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec)
                # Add a small softening factor to avoid division by zero if bodies get too close
                force_mag = G * masses[i] * masses[j] / (r_mag**2 + 1e-9)
                force_vec = force_mag * r_vec / r_mag
                forces[i] += force_vec
    return forces

def calculate_energy(masses, positions, velocities):
    """Calculates the total kinetic and potential energy of the system."""
    # Kinetic Energy: 0.5 * m * v^2
    ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)

    # Potential Energy: -G * m1 * m2 / r
    pe = 0
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[j] - positions[i]
            r_mag = np.linalg.norm(r_vec)
            pe -= G * masses[i] * masses[j] / r_mag
    return ke + pe


# --- Step 3: Simulation Loop (Leapfrog Integration) ---

# Initial force and acceleration calculation
forces = calculate_forces(masses, positions)
accelerations = forces / masses[:, np.newaxis]

# Leapfrog integration
for i in range(num_steps):
    # Store data for this step
    positions_history[i] = positions
    energy_history[i] = calculate_energy(masses, positions, velocities)

    # "Kick" (update velocities by half a time step)
    velocities += accelerations * dt / 2

    # "Drift" (update positions by a full time step)
    positions += velocities * dt

    # Update forces and accelerations at the new positions
    forces = calculate_forces(masses, positions)
    accelerations = forces / masses[:, np.newaxis]

    # "Kick" (update velocities by another half a time step)
    velocities += accelerations * dt / 2


# --- Step 4: Visualization ---

# Create the animation
fig_anim, ax_anim = plt.subplots()
ax_anim.set_aspect('equal')
ax_anim.set_xlim(-4.0e11, 4.0e11)
ax_anim.set_ylim(-4.0e11, 4.0e11)
ax_anim.set_title("N-Body Simulation (Leapfrog Integration)")
ax_anim.set_xlabel("x (m)")
ax_anim.set_ylabel("y (m)")
ax_anim.grid(True)

# Create plot objects for bodies and their trails
bodies = [ax_anim.plot([], [], 'o', markersize=ms)[0] for ms in [10, 5, 4, 3]]
trails = [ax_anim.plot([], [], '-', linewidth=1)[0] for _ in range(N)]

def animate(frame):
    """Animation function."""
    for i in range(N):
        bodies[i].set_data([positions_history[frame, i, 0]], [positions_history[frame, i, 1]])
        trails[i].set_data(positions_history[:frame+1, i, 0], positions_history[:frame+1, i, 1])
    return bodies + trails

ani = FuncAnimation(fig_anim, animate, frames=num_steps, interval=20, blit=True)
print("Saving animation... this may take a moment.")
ani.save('n_body_simulation.gif', writer='pillow', fps=60)
print("Animation saved as n_body_simulation.gif")

# Create the energy plot
fig_energy, ax_energy = plt.subplots()
ax_energy.plot(np.arange(num_steps) * dt / (365 * 24 * 3600), energy_history)
ax_energy.set_title("Total System Energy Over Time")
ax_energy.set_xlabel("Time (years)")
ax_energy.set_ylabel("Total Energy (Joules)")
ax_energy.grid(True)

print("Saving energy conservation plot...")
fig_energy.savefig("energy_conservation.png")
print("Energy plot saved as energy_conservation.png")