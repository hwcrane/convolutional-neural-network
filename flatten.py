import numpy as np
from layers import Layer


class Flatten(Layer):
    def __init__(self) -> None:
        self.input_shape = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        if self.input_shape == None:
            self.input_shape = input.shape[1:]
            self.output_size = np.prod(self.input_shape)

        return input.reshape(-1, self.output_size)

    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        # Reshapes derivatives to the original input shape
        return derivatives.reshape(-1, *self.input_shape)
