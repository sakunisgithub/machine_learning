import numpy as np
from matplotlib import pyplot as plt

X = np.array([2, 0, 2, 2.1])
Y = np.array([0.95, 0.5, 0.10, 0.099])

# creating the error surface
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
# plt.contour(W, B, error, levels=20, colors='black', linewidths=0.5)
ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f", colors = "black")

plt.colorbar(contour, label='Error')

plt.xlabel('W')

plt.ylabel('B')

# implementing gradient descent
def dw(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) * X ) )

def db(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) ) )

target_error = float(input("Input target error, for example 0.48 : "))

def gradient_descent(X, Y, w_initial, b_initial, eta, max_epochs, target_error) :

    w, b = w_initial, b_initial

    # to store w and b updated values
    W, B = np.zeros(max_epochs + 1), np.zeros(max_epochs + 1)

    W[0] = w_initial
    B[0] = b_initial

    epoch_required = reached_error = 0

    for i in range(max_epochs) :
        w = w - eta * dw(X, Y, w, b)
        b = b - eta * db(X, Y, w, b)

        W[i+1] = w
        B[i+1] = b

        e = np.sum( (Y - f(X, w, b))**2 )

        if e <= target_error and reached_error == 0:
            epoch_required = i+1
            reached_error = 1
        elif i == max_epochs - 1 and reached_error == 0:
            epoch_required = -1            

    return((w, b, W, B, epoch_required))

w, b, W_values, B_values, epoch_required = gradient_descent(X, Y, -1, 4, 0.5, 3000, target_error)

print(f"Epochs required for Vanilla GD to reach error {target_error} is {epoch_required}, Error at epoch {epoch_required} is {np.sum( (Y - f(X, W_values[epoch_required], B_values[epoch_required]))**2 )}")

def momentum_gradient_descent(X, Y, w_initial, b_initial, eta, max_epochs, gamma, target_error) :

    w, b = w_initial, b_initial

    # to store w and b updated values
    W, B = np.zeros(max_epochs + 1), np.zeros(max_epochs + 1)

    W[0] = w_initial
    B[0] = b_initial
    
    previous_update_w = 0
    previous_update_b = 0

    epoch_required = reached_error = 0

    for i in range(max_epochs) :

        current_update_w = gamma * previous_update_w + eta * dw(X, Y, w, b) 
        current_update_b = gamma * previous_update_b + eta * db(X, Y, w, b) 
        
        w = w - current_update_w 
        b = b - current_update_b 

        W[i+1] = w
        B[i+1] = b

        previous_update_w = current_update_w 
        previous_update_b = current_update_b 

        e = np.sum( (Y - f(X, w, b))**2 )

        if e <= target_error and reached_error == 0:
            epoch_required = i+1
            reached_error = 1
        elif i == max_epochs - 1 and reached_error == 0:
            epoch_required = -1            

    return((w, b, W, B, epoch_required))

w, b, W_values_momentum_GD, B_values_momentum_GD, epoch_required = momentum_gradient_descent(X, Y, -1, 4, 0.5, 3000, 0.9, target_error)

print(f"Epochs required for Momentum GD to reach error {target_error} is {epoch_required}, Error at epoch {epoch_required} is {np.sum( (Y - f(X, W_values_momentum_GD[epoch_required], B_values_momentum_GD[epoch_required]))**2 )}")

def nestrov_accelerated_gradient_descent(X, Y, w_initial, b_initial, eta, max_epochs, gamma, target_error) :

    w, b = w_initial, b_initial

    W, B = np.zeros(max_epochs + 1), np.zeros(max_epochs + 1)

    W[0] = w_initial
    B[0] = b_initial
    
    previous_update_w = 0
    previous_update_b = 0

    reached_error = epoch_required = 0

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

        e = np.sum( (Y - f(X, w, b))**2 )

        if e <= target_error and reached_error == 0:
            epoch_required = i+1
            reached_error = 1
        elif i == max_epochs - 1 and reached_error == 0:
            epoch_required = -1            

    return((w, b, W, B, epoch_required))

w, b, W_values_NAGD, B_values_NAGD, epoch_required = nestrov_accelerated_gradient_descent(X, Y, -1, 4, 0.5, 3000, 0.9, target_error)

print(f"Epochs required for Nestrov GD to reach error {target_error} is {epoch_required}, Error at epoch {epoch_required} is {np.sum( (Y - f(X, W_values_NAGD[epoch_required], B_values_NAGD[epoch_required]))**2 )}")

plt.ion()

for i in range(len(W_values)-1) :
    ax.quiver(
        W_values[i], B_values[i],              # starting points
        W_values[i+1] - W_values[i], B_values[i+1] - B_values[i],
        angles='xy', scale_units='xy', scale=1, 
        color='red', width=0.005)

    ax.quiver(
        W_values_momentum_GD[i], B_values_momentum_GD[i],              # starting points
        W_values_momentum_GD[i+1] - W_values_momentum_GD[i], B_values_momentum_GD[i+1] - B_values_momentum_GD[i],
        angles='xy', scale_units='xy', scale=1, 
        color='blue', width=0.005)

    ax.quiver(
        W_values_NAGD[i], B_values_NAGD[i],              # starting points
        W_values_NAGD[i+1] - W_values_NAGD[i], B_values_NAGD[i+1] - B_values_NAGD[i],
        angles='xy', scale_units='xy', scale=1, 
        color='black', width=0.005)

    plt.title(f"Vanilla GD vs Momentum GD vs Nestrov GD at epoch = {i}")

    plt.pause(0.3)

    if not plt.fignum_exists(1) :
        break

plt.show()