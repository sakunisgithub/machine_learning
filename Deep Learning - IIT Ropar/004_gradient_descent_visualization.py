import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# data
X = np.array([0.5, 2.5])
Y = np.array([0.2, 0.9])

# creating the error surface
w = np.linspace(-6, 6, 20)
b = np.linspace(-6, 6, 20)

W, B = np.meshgrid(w, b)

def f(x, w, b):
    return(1/(1 + np.exp(-(w*x + b))))

error = np.zeros_like(W)

for i in range(W.shape[0]) :
    for j in range(W.shape[1]) :
        error[i, j] = np.sum( (Y - f(X, W[i, j], B[i, j]))**2 )

def E(X, Y, w, b) :
   return(np.sum( (Y - f(X, w, b))**2) )
   
ax = plt.axes(projection = "3d")

ax.plot_surface(W, B, error, cmap = "coolwarm", alpha = 0.8)

ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("Error")

ax.set_zlim(-1.5, 2)

# implementing gradient descent
def dw(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) * X ) )

def db(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) ) )

def gradient_descent(X, Y, w_initial, b_initial, eta, max_epochs) :

    w, b = w_initial, b_initial

    # to store w and b updated values
    W, B = np.zeros(max_epochs + 1), np.zeros(max_epochs + 1)

    W[0] = w_initial
    B[0] = b_initial

    for i in range(max_epochs) :
        w = w - eta * dw(X, Y, w, b)
        b = b - eta * db(X, Y, w, b)

        W[i+1] = w
        B[i+1] = b

    return((w, b, W, B))

w, b, W_values, B_values = gradient_descent(X, Y, -2, -2, 1, 1000)

print(f"w = {w}, b = {b}")

for i in range(len(W_values)) :
    ax.scatter(W_values[i], B_values[i], E(X, Y, W_values[i], B_values[i]), color = "red", s = 5, label = f"{i}")
    ax.scatter(W_values[i], B_values[i], -1.5, color = "black", s = 5)

    ax.set_title(f"epoch = {i}")

    plt.pause(0.3)

plt.show()

