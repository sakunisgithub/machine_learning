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
   
contour = plt.contourf(W, B, error, levels=20, cmap='coolwarm', alpha = 0.8)

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

w, b, W_values_batch_GD, B_values_batch_GD = gradient_descent(X, Y, -2, -2, 1, 1000)

def momentum_gradient_descent(X, Y, w_initial, b_initial, eta, max_epochs, gamma) :

    w, b = w_initial, b_initial

    # to store w and b updated values
    W, B = np.zeros(max_epochs + 1), np.zeros(max_epochs + 1)

    W[0] = w_initial
    B[0] = b_initial
    
    previous_update_w = 0
    previous_update_b = 0

    for i in range(max_epochs) :

        current_update_w = gamma * previous_update_w + eta * dw(X, Y, w, b) 
        current_update_b = gamma * previous_update_b + eta * db(X, Y, w, b) 
        
        w = w - current_update_w 
        b = b - current_update_b 

        W[i+1] = w
        B[i+1] = b

        previous_update_w = current_update_w 
        previous_update_b = current_update_b 

    return((w, b, W, B))

w, b, W_values_momentum_GD, B_values_momentum_GD = momentum_gradient_descent(X, Y, -2, -2, 1, 1000, 0.9)

def nestrov_accelerated_gradient_descent(X, Y, w_initial, b_initial, eta, max_epochs, gamma) :

    w, b = w_initial, b_initial

    # to store w and b updated values
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

w, b, W_values_NAGD, B_values_NAGD = nestrov_accelerated_gradient_descent(X, Y, -2, -2, 1, 1000, 0.9)

print(f"w = {w}, b = {b}")

plt.ion()

for i in range(len(W_values_batch_GD)) :
    plt.scatter(W_values_batch_GD[i], B_values_batch_GD[i], color = "black", s = 5, label = f"{i}")

    plt.scatter(W_values_momentum_GD[i], B_values_momentum_GD[i], color = "blue", s = 5, label = f"{i}")

    plt.scatter(W_values_NAGD[i], B_values_NAGD[i], color = "red", s = 5, label = f"{i}")


    plt.title(f"epoch = {i}")

    plt.pause(0.3)

    if not plt.fignum_exists(1) :
        break

plt.show()