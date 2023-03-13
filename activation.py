import enum
from typing import List
from layers import Layer
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return max(x, 0)

def relu_derivative(x):
    return 0 if x < 0 else x

class Functions(enum.Enum):
    SIGMOID = (sigmoid, sigmoid_derivative)
    RELU = (relu, relu_derivative)

class Activation(Layer):
    def __init__(self, function: Functions) -> None:
        super().__init__()
        self.function = np.vectorize(function.value[0])
        self.derivitive = np.vectorize(function.value[1])
        self.output_shape: List[int] = []

    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.function(input)

    def backward(self, derivitive: np.ndarray) -> np.ndarray:
        return self.derivitive(derivitive)

    def get_output_shape(self) -> List[int]:
        return self.output_shape

    def compile(self, input_shape: List[int]):
        self.output_shape = input_shape
        self.initalised = True
