import numpy as np
from typing import List
from layers import Layer

class Dense(Layer):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.output_shape: List[int] = [-1, size]
        self.weights = np.array([])
        self.biases = np.array([])
        self.input = np.array([])
        self.output = np.array([])

    def compile(self, input_shape: List[int]):
        self.initalised = True
        self.weights = np.random.normal(0, 1, size=[input_shape[1], self.output_shape[1]])
        self.biases = np.random.normal(0, 1, size=[1, self.output_shape[1]])

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        ones = np.ones(input.shape[0]).reshape(-1, 1)
        self.output = input@self.weights + ones@self.biases
        return self.output

    def backward(self, derivatives: np.ndarray) -> np.ndarray:
        return super().backward(derivatives)

    def get_output_shape(self) -> List[int]:
        return self.output_shape
