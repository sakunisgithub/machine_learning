import numpy as np

X = np.array([0.5, 2.5])
Y = np.array([0.2, 0.9])

# prediction equation -- sigmoid function

def f(x, w, b) :
    return(1/(1 + np.exp(-(w*x + b))))

# derivatives of Squared Error Loss function w.r.t. the parameters

def dw(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) * X ) )

def db(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) ) )

# gradient descent

def gradient_descent(X, Y, w_initial, b_initial, eta, max_epochs) :

    w, b = w_initial, b_initial

    for i in range(max_epochs) :
        w = w - eta * dw(X, Y, w, b)
        b = b - eta * db(X, Y, w, b)

    return((w, b))

# doing gradient descent

w, b = gradient_descent(X, Y, 1, 1, 1, 1000)

print(f"w = {w}, b = {b}")