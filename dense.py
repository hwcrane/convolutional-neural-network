import numpy as np
from layers import Layer

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size

        # Uses Xavier initialization for the weights
        fan_in = input_size
        fan_out = output_size
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.weights = np.random.uniform(-limit, limit, size=[input_size, output_size])

        # Initialize biases to zero
        self.biases = np.zeros((1, output_size))

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        # Compute the gradients with respect to the weights, inputs, and biases
        dW = np.dot(self.input.T, derivatives)
        dX = np.dot(derivatives, self.weights.T)
        dB = np.sum(derivatives, axis=0, keepdims=True)

        # Update the weights and biases using gradient descent
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * dB

        return dX

