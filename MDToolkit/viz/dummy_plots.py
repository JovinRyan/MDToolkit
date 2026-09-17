import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0
a = 1.5
sigma = 0.25

# x-axis
x = np.linspace(0, 2.5 * T, 1000)

# Distribution centered at T
y = np.exp(-0.5 * ((x - T) / sigma)**2)

# Plot
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2)

# Highlight distribution above aT
mask = x >= a * T
ax.fill_between(
    x[mask],
    y[mask],
    alpha=0.3
)

# Mark aT
ax.axvline(
    a * T,
    linestyle="--",
    linewidth=1.5,
    label=r"$aT$"
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$P(x)$")
ax.legend()

plt.tight_layout()
plt.show()