from functools import reduce
import numpy as np
from layers import Layer
from typing import List


class Flattern(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.output_shape: List[int] = []
        self.input_shape = []

    def compile(self, input_shape: List[int]):
        self.output_shape = [-1, reduce(lambda x, y: x * y, input_shape)]
        self.initalised = True

    def forward(self, input: np.ndarray) -> np.ndarray:
        return input.reshape(*self.output_shape)

    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        return super().backward(derivatives, learning_rate)

    def get_output_shape(self) -> List[int]:
        return self.output_shape
