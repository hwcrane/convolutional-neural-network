from layers import Layer
import numpy as np
import numba


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
