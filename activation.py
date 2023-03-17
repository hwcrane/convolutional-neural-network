import enum
from layers import Layer
import numpy as np
import numba

#
# @numba.njit()
# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))
#
#
# @numba.njit()
# def sigmoid_derivative(x):
#     return sigmoid(x) * (1 - sigmoid(x))
#
#
# @numba.njit()
# def relu(x):
#     return np.maximum(x, 0)
#
#
# @numba.njit()
# def relu_derivative(x):
#     return 1 if x > 0 else 0
#
#
# class Functions(enum.Enum):
#     SIGMOID = (sigmoid, sigmoid_derivative)
#     RELU = (relu, relu_derivative)
#
#
# class Activation(Layer):
#     def __init__(self, function: Functions) -> None:
#         self.input = np.array([])
#         self.function = np.vectorize(function.value[0])
#         self.derivitive = np.vectorize(function.value[1])
#
#     def forward(self, input: np.ndarray) -> np.ndarray:
#         self.input = input
#         return self.function(self.input)
#
#     def backward(self, derivitive: np.ndarray, learning_rate) -> np.ndarray:
#         return self.derivitive(self.input) * derivitive

@numba.njit()
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@numba.njit()
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


@numba.njit()
def relu(x):
    return np.maximum(x, 0)


@numba.njit()
def relu_derivative(x):
    return 1.0 * (x > 0)


class Activation(Layer):
    def __init__(self, function: str) -> None:
        self.input = np.array([])
        self.function = function.lower()

        if self.function == "sigmoid":
            self.activation_func = sigmoid
            self.derivative = sigmoid_derivative
        elif self.function == "relu":
            self.activation_func = relu
            self.derivative = relu_derivative
        else:
            raise ValueError(f"Invalid activation function: {function}")

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return self.activation_func(self.input)

    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        return self.derivative(self.input) * derivatives

