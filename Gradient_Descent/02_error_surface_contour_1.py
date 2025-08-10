import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

X = np.array([1, -3.5])
Y = np.array([0.2, 0.9])

w = np.linspace(-5, 5, 256)
b = np.linspace(-5, 5, 256)

W, B = np.meshgrid(w, b)

def f(x, w, b):
    return(1/(1 + np.exp(-(w*x + b))))

error = np.zeros_like(W)

for i in range(W.shape[0]) :
    for j in range(W.shape[1]) :
        error[i, j] = np.sum( (Y - f(X, W[i, j], B[i, j]))**2 )

# low_levels = np.linspace(error.min(), 0.1, 20)   # dense near low
# high_levels = np.linspace(0.3, error.max(), 5)   # sparse in high
# levels = np.unique(np.concatenate([low_levels, high_levels]))

fig, ax = plt.subplots()
contour = ax.contourf(W, B, error, levels=20, cmap='jet', alpha = 0.8)
plt.contour(W, B, error, levels=20, colors='black', linewidths=0.5)
ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f", colors = "black")

plt.colorbar(contour, label='Error')

plt.xlabel('W')
plt.ylabel('B')

plt.scatter(-0.8, -0.6, color = 'red') # the solution i.e. the point of minima

plt.show()