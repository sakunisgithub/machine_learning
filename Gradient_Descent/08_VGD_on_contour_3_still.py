import numpy as np
from matplotlib import pyplot as plt

X = np.array([2, 0, 2, 2.1])
Y = np.array([0.95, 0.5, 0.10, 0.099])

w = np.linspace(-5, 5, 256)
b = np.linspace(-5, 5, 256)

W, B = np.meshgrid(w, b)

def f(x, w, b):
    return(1/(1 + np.exp(-(w*x + b))))

error = np.zeros_like(W)

for i in range(W.shape[0]) :
    for j in range(W.shape[1]) :
        error[i, j] = np.sum( (Y - f(X, W[i, j], B[i, j]))**2 )

fig, ax = plt.subplots()
contour = ax.contourf(W, B, error, levels=20, cmap='jet', alpha = 0.7)
plt.contour(W, B, error, levels=20, colors='black', linewidths=0.5)
ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f", colors = "black")

plt.colorbar(contour, label='Error')

plt.xlabel('W')
plt.ylabel('B')

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

def take_input() :
    print("Provide initial values of (w, b), for example (-1, 4)")
    w_initial = float(input("Give w_initial = "))
    b_initial = float(input("Give b_initial = "))

    print("Provide learning rate, for example 1, 0.5, 0.1, 0.01")
    learning_rate = float(input("Give learning rate (eta) = "))

    return((w_initial, b_initial, learning_rate))

w_init, b_init, eta = take_input()

w, b, W_values, B_values = gradient_descent(X, Y, w_init, b_init, eta, 1000)

print(f"w = {w}, b = {b}")

ax.quiver(
    W_values[:-1], B_values[:-1],              # starting points
    np.diff(W_values), np.diff(B_values),      # directions
    angles='xy', scale_units='xy', scale=1, 
    color='black', width=0.005)

plt.show()