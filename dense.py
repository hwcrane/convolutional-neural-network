import numba
import numpy as np
from layers import Layer

# Jit compiled funtions for forwards and backwards propagation for increased speed


@numba.njit()
def dense_forward(input, weights, biases):
    return np.dot(input, weights) + biases


@numba.njit()
def dense_backward(input, derivatives, weights):
    dW = np.dot(input.T, derivatives)
    dX = np.dot(derivatives, weights.T)
    dB = np.sum(derivatives, axis=0)

    return dW, dX, dB


class Dense(Layer):
    def __init__(self, output_size: int) -> None:
        self.input_size = None
        self.output_size = output_size

    def initialise(self, input_size):
        self.input_size = input_size
        # Uses Xavier initialisation for the weights
        limit = np.sqrt(6 / (self.input_size + self.output_size))
        self.weights = np.random.uniform(
            -limit, limit, size=[self.input_size, self.output_size]
        )

        # Initialize biases to zero
        self.biases = np.zeros((1, self.output_size))

    def forward(self, input: np.ndarray) -> np.ndarray:
        if self.input_size == None:
            self.initialise(input.shape[1])

        self.input = input
        return dense_forward(input, self.weights, self.biases)

    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        # Compute the gradients with respect to the weights, inputs, and biases
        dW, dX, dB = dense_backward(self.input, derivatives, self.weights)

        # Update the weights and biases using gradient descent
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * dB

        return dX
