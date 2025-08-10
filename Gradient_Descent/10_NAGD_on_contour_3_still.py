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

contour = ax.contourf(W, B, error, levels=20, cmap='jet', alpha = 0.5)

plt.colorbar(contour, label='Error')

plt.xlabel('W')

plt.ylabel('B')

# implementing gradient descent
def dw(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) * X ) )

def db(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) ) )

def nestrov_accelerated_gradient_descent(X, Y, w_initial, b_initial, eta, max_epochs, gamma) :

    w, b = w_initial, b_initial

    W, B = np.zeros(max_epochs + 1), np.zeros(max_epochs + 1)

    W[0] = w_initial
    B[0] = b_initial
    
    previous_update_w = 0
    previous_update_b = 0

    for i in range(max_epochs) :

        w_look_ahead = w - gamma * previous_update_w
        b_look_ahead = b - gamma * previous_update_b

        current_update_w = gamma * previous_update_w + eta * dw(X, Y, w_look_ahead, b_look_ahead) 
        current_update_b = gamma * previous_update_b + eta * db(X, Y, w_look_ahead, b_look_ahead) 
        
        w = w - current_update_w 
        b = b - current_update_b 

        W[i+1] = w
        B[i+1] = b

        previous_update_w = current_update_w 
        previous_update_b = current_update_b 

    return((w, b, W, B))

w, b, W_values_NAGD, B_values_NAGD = nestrov_accelerated_gradient_descent(X, Y, -1, 4, 0.5, 1000, 0.9)

print(f"w = {w}, b = {b}")

ax.quiver(
    W_values_NAGD[:-1], B_values_NAGD[:-1],              # starting points
    np.diff(W_values_NAGD), np.diff(B_values_NAGD),      # directions
    angles='xy', scale_units='xy', scale=1, 
    color='black', width=0.005)

plt.show()