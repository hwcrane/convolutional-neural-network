from layers import Layer
import numpy as np
from typing import List


class Convolutional(Layer):
    def __init__(self, filter_sizes: List[int], num_filters: int) -> None:
        super().__init__()

    def compile(self, input_shape: List[int]):
        return super().compile(input_shape)

    def forward(self, input: np.ndarray) -> np.ndarray:
        return super().forward(input)

    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        return super().backward(derivatives, learning_rate)

    def get_output_shape(self) -> List[int]:
        return super().get_output_shape()
