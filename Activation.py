import numpy as np


def activate(x, fun='linear'):
    if fun.lower() == 'sigmoid':
        return __sigmoid(x)
    elif fun.lower() == 'sigmoid_der':
        return __sigmoid_derivative(x)
    elif fun.lower() == 'relu':
        return __relu(x)
    elif fun.lower() == 'relu_der':
        return __relu_derivative(x)
    elif fun.lower() == 'linear':
        return __linear(x)
    elif fun.lower() == 'tanh':
        return __tanh(x)
    elif fun.lower() == 'tanh_der':
        return __tanh_derivative(x)
    elif fun.lower() == 'elu':
        return __elu(x)
    else:
        raise ValueError('Invalid Activation Function !')


# activation functions
def __sigmoid(x):
    y = 1 / (1 + np.exp(-1 * x))
    return y


def __relu(x):
    if np.isscalar(x):
        y = np.max((x, 0))
    else:
        zero_aux = np.zeros(x.shape)
        meta_x = np.stack((x, zero_aux), axis=-1)
        y = np.max(meta_x, axis=-1)
    return y


def __linear(x):
    y = np.linspace(-10, 10)
    x = y
    return y


def __tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def __elu(x):
    if x > 0:
        return x
    else:
        return np.exp(x)-1


# Backpropagation ( Derivative Functions )
def __sigmoid_derivative(x):
    return x * (1 - x)


def __relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def __tanh_derivative(x):
    return 1 - np.power(x, 2)
