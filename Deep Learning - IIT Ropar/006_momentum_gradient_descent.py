import numpy as np

X = np.array([0.5, 2.5])
Y = np.array([0.2, 0.9])

def f(x, w, b) :
    return(1/(1 + np.exp(-(w*x + b))))

def dw(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) * X ) )

def db(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) ) )

def momentum_gradient_descent(X, Y, w_initial, b_initial, eta, max_epochs, gamma) :

    w, b = w_initial, b_initial
    
    previous_update_w = 0
    previous_update_b = 0

    for i in range(max_epochs) :

        current_update_w = gamma * previous_update_w + eta * dw(X, Y, w, b) 
        current_update_b = gamma * previous_update_b + eta * db(X, Y, w, b) 
        
        w = w - current_update_w 
        b = b - current_update_b 

        previous_update_w = current_update_w 
        previous_update_b = current_update_b 

    return((w, b))

w, b = momentum_gradient_descent(X, Y, 1, 1, 1, 1000, 0.9)

print(f"w = {w}, b = {b}")