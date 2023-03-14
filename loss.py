import numpy as np


def mean_squared_err(actual, predicted):
    return np.mean(np.power(actual - predicted, 2))


def mean_squared_err_deriv(actual, predicted):
    return 2 * (predicted - actual) / actual.size
