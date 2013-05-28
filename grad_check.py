import numpy as np

epsilon = 10**-4

def calc_grad_numeric(cost_func, params, data):
    grad = np.zeros(params.size)
    for i in range(params.size):
        params1 = params.copy()
        params1[i] -= epsilon
        params2 = params.copy()
        params2[i] += epsilon
        y1 = cost_func(params1, data)        
        y2 = cost_func(params2, data)
        grad[i] = (y2 - y1) / 2. / epsilon

    return grad
