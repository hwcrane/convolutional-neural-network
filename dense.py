import numpy as np
from typing import List
from layers import Layer


class Dense(Layer):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.output_shape: List[int] = [-1, size]
        self.weights = np.array([])
        self.biases = np.array([])
        self.output = np.array([])

    def compile(self, input_shape: List[int]):
        self.initalised = True
        self.weights = np.random.normal(
            0, 1, size=[input_shape[1], self.output_shape[1]]
        )
        self.biases = np.random.normal(0, 1, size=[1, self.output_shape[1]])

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        ones = np.ones(input.shape[0]).reshape(-1, 1)
        self.output = self.input @ self.weights + ones @ self.biases
        return self.output

    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        dW = np.dot(self.input.T, derivatives)
        dX = np.dot(derivatives, self.weights.T)
        dB = np.sum(derivatives, axis=0).reshape(1, -1)

        self.weights -= dW * learning_rate
        self.biases -= dB * learning_rate

        return dX

    def get_output_shape(self) -> List[int]:
        return self.output_shape
