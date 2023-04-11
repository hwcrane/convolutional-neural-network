from layers import Layer
from numpy.lib.stride_tricks import as_strided
from scipy.signal import convolve2d
import numpy as np
import numba

# numba.njit()
def conv2d(a, b):
    Hout = a.shape[1] - b.shape[0] + 1
    Wout = a.shape[2] - b.shape[1] + 1

    a = as_strided(a, (a.shape[0], Hout, Wout, b.shape[0], b.shape[1], a.shape[3]), a.strides[:3] + a.strides[1:])

    return np.tensordot(a, b, axes=3)

class Conv2D(Layer):
    def __init__(
        self,
        num_channels: int,  # Number of input channels (3 for RBB)
        num_filters: int,  # Number of filters
        filter_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(
            filter_size, filter_size, num_channels, num_filters
        )

        self.biases = np.zeros(num_filters)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.last_input = input
        return conv2d(input, self.weights)


    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        # Extract the dimensions of the input tensor
        batches, input_height, input_width, channels = self.last_input.shape

        # Compute the output dimensions based on the layer's parameters
        output_height = (
            input_height + 2 * self.padding - self.filter_size
        ) // self.stride + 1
        output_width = (
            input_width + 2 * self.padding - self.filter_size
        ) // self.stride + 1

        # Add padding to the input tensor using `np.pad`
        input_padded = np.pad(
            self.last_input,
            (
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            ),
            mode="constant",
        )

        # Creates a "view" tensor of the input tensor
        # This creates a sliding window of size `(filter_size, filter_size, channels)` over the input tensor
        # The result is a tensor of shape `(batch_size, output_height, output_width, filter_size, filter_size, channels)`
        input_strides = as_strided(
            input_padded,
            (
                batches,
                output_height,
                output_width,
                self.filter_size,
                self.filter_size,
                channels,
            ),
            input_padded.strides[:3]
            + (
                self.stride * input_padded.strides[1],
                self.stride * input_padded.strides[2],
            )
            + input_padded.strides[3:],
        )

        # Compute the gradients of the weights and biases
        dweights = np.tensordot(input_strides, derivatives, axes=((0, 1, 2), (0, 1, 2)))
        dbiases = np.sum(derivatives, axis=(0, 1, 2))

        # Update the weights and biases
        self.weights -= learning_rate * dweights
        self.biases -= learning_rate * dbiases

