import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

X = np.array([0.5, 2.5])
Y = np.array([0.2, 0.9])

w = np.linspace(-6, 6, 20)
b = np.linspace(-6, 6, 20)

W, B = np.meshgrid(w, b)

def f(x, w, b):
    return(1/(1 + np.exp(-(w*x + b))))

error = np.zeros_like(W)

for i in range(W.shape[0]) :
    for j in range(W.shape[1]) :
        error[i, j] = np.sum( (Y - f(X, W[i, j], B[i, j]))**2 )

ax = plt.axes(projection = "3d")

ax.plot_surface(W, B, error, cmap = "coolwarm")

ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("Error")

ax.set_zlim(-2, 2)

plt.show()
