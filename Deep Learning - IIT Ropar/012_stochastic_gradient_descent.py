import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/sakunisgithub/data_sets/refs/heads/master/SGD_data.csv')

X = df.iloc[:, 0].to_numpy()
Y = df.iloc[:, 1].to_numpy() 

def f(x, w, b) :
    return(1/(1 + np.exp(-(w*x + b))))

def dw(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) * X ) )

def db(X, Y, w, b) :
    return( np.sum( (f(X, w, b) - Y) * f(X, w, b) * (1 - f(X, w, b)) ) )

def stochastic_gradient_descent(X, Y, w_initial, b_initial, eta, max_epochs) :

    w, b = w_initial, b_initial

    for i in range(max_epochs) :
        for j in range(len(X)) :
            w = w - eta * dw(X[j], Y[j], w, b)
            b = b - eta * db(X[j], Y[j], w, b)

    return((w, b))

w, b = stochastic_gradient_descent(X, Y, 1, 1, 1, 1000)

print(f"w = {w}, b = {b}")