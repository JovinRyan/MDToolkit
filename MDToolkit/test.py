import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from MDToolkit.math.smoothing import lowess

# 1. Generate noisy, non-linear dummy data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.4, size=len(x))


# 3. Extract the smoothed coordinates
# The function returns a 2D numpy array of shape (N, 2) representing [x_sorted, y_smoothed]
x_smooth, y_smooth = lowess(x, y, 25)

# 4. Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='lightgray', label='Noisy Data')
plt.plot(x_smooth, y_smooth, color='red', linewidth=2, label='LOWESS Fit (frac=0.25)')
plt.title('LOWESS Smoothing Example')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
