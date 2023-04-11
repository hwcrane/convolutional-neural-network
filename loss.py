import numpy as np
import numba


@numba.njit()
def mean_squared_err(actual, predicted):
    return np.mean(np.power(actual - predicted, 2))


@numba.njit()
def mean_squared_err_deriv(actual, predicted):
    return 2 * (predicted - actual) / actual.size
