import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
sheet_size = 30.0          # Size of plotted square
supercell_factor = 3       # Generate a lattice this many times larger
a = 1.0
pore_radius = 5.0

n_sampling_points = 20
sampling_power = 0.6       # Smaller -> more points near pore

# -----------------------------
# Generate large graphene lattice
# -----------------------------
a1 = np.array([np.sqrt(3) * a, 0])
a2 = np.array([np.sqrt(3) / 2 * a, 1.5 * a])

basis = [
    np.array([0.0, 0.0]),
    np.array([np.sqrt(3)/2 * a, 0.5*a]),
]

supercell_size = sheet_size * supercell_factor

nx = int(np.ceil(supercell_size / np.linalg.norm(a1))) + 2
ny = int(np.ceil(supercell_size / 1.5)) + 2

points = []

for i in range(nx):
    for j in range(ny):
        R = i*a1 + j*a2
        for b in basis:
            points.append(R + b)

points = np.asarray(points)

# Center lattice
points -= np.mean(points, axis=0)

# Remove pore
r = np.linalg.norm(points, axis=1)
points = points[r > pore_radius]

# Clip to plotting region
half = sheet_size / 2

mask = (
    (np.abs(points[:,0]) <= half) &
    (np.abs(points[:,1]) <= half)
)

points = points[mask]

# -----------------------------
# Sampling points
# -----------------------------
origin = np.array([-half, -half])

# Cluster points toward the pore
t = np.linspace(0, 1, n_sampling_points)
t = t**sampling_power

probe = origin[None, :] * (1 - t[:, None])

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(6.5,6.5))

ax.scatter(
    points[:,0],
    points[:,1],
    s=12,
    color="black",
    zorder=1
)

ax.scatter(
    probe[:,0],
    probe[:,1],
    s=55,
    color="crimson",
    edgecolor="white",
    linewidth=0.7,
    zorder=3,
    label="Sampling positions"
)

theta = np.linspace(0, 2*np.pi, 400)
ax.plot(
    pore_radius*np.cos(theta),
    pore_radius*np.sin(theta),
    "--",
    color="steelblue",
    lw=2,
    label="Pore"
)

ax.set_xlim(-half, half)
ax.set_ylim(-half, half)
ax.set_aspect("equal")

ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

legend = ax.legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=1.0,
    fancybox=False
)

legend.get_frame().set_linewidth(0.8)
plt.tight_layout()
plt.show()

# -----------------------------
# Dummy electron wind force
# -----------------------------
distance = np.linalg.norm(probe, axis=1)

F_min = 0.4      # force at pore center
F_max = 3.0      # bulk saturation value

transition = pore_radius + 3.0   # midpoint of sigmoid (Å)
width = 1.5                      # controls steepness (Å)

electron_wind_force = (
    F_min
    + (F_max - F_min)
    / (1 + np.exp(-(distance - transition) / width))
)
# -----------------------------
# Dummy electron wind force plot
# -----------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.plot(
    distance,
    electron_wind_force,
    "-o",
    lw=2,
    ms=5
)

ax.set_xlabel("Distance from Pore Center (Å)")
ax.set_ylabel("Electron Wind Force (arb. units)")
ax.set_title("Dummy Electron Wind Force Profile")

ax.grid(alpha=0.3)

# Optional: indicate pore edge
ax.axvline(
    pore_radius,
    color="steelblue",
    linestyle="--",
    label="Pore edge"
)

ax.legend()

plt.tight_layout()
plt.show()