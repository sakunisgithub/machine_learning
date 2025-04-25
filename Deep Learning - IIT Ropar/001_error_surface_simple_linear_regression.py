import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# True model parameters
true_w = 2
true_b = -1.5

# Generate x values
x = np.random.random(100) 

# Generate y values with some noise
noise = np.random.normal(0, 1, len(x))
y = true_w * x + true_b + noise

# Define a range of w values to compute the error surface
w_values = np.linspace(-3, 3, 100)
b_values = np.linspace(-3, 3, 100)

# Compute the error surface (SSE)
W, B = np.meshgrid(w_values, b_values)
SSE = np.zeros_like(W)

for i in range(len(w_values)):
    for j in range(len(b_values)):
        y_pred = W[i, j] * x + B[i, j]
        SSE[i, j] = np.sum((y - y_pred) ** 2)

# Plot the error surface
ax = plt.axes(projection = "3d")

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('error')

ax.plot_surface(W, B, SSE, cmap = 'coolwarm', alpha = 0.8)
#ax.scatter(true_w, true_b, np.sum((y - true_w * x - true_b) ** 2), color = "red", s = 30)
#ax.scatter(true_w, true_b, 0, color = "black", s = 30)
plt.show()