import numpy as np
from layers import Layer
from typing import List


class Flatten(Layer):
    def __init__(self, input_shape: List[int]) -> None:
        self.input_shape = input_shape
        self.output_size = np.prod(input_shape)

    def forward(self, input: np.ndarray) -> np.ndarray:
        return input.reshape(-1, self.output_size)

    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        # Reshapes derivatives to the original input shape
        return derivatives.reshape(-1, *self.input_shape)

