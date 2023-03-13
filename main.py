from typing import List
from keras.utils.data_utils import time
import numpy as np
from activation import Activation, Functions
from dense import Dense
from layers import Layer
from flattern import Flattern
from keras.datasets import mnist
import matplotlib.pyplot as plt

class Network:
    def __init__(self, layers: List[Layer] = []) -> None:
        self.layers = layers

    def compile(self, input_shape: List[int]):
        last_shape = input_shape
        for layer in self.layers:
            layer.compile(last_shape)
            last_shape = layer.get_output_shape()

    def forward(self, input: np.ndarray) -> np.ndarray:
        [input := layer.forward(input) for layer in self.layers]
        # for layer in self.layers:
        #     input = layer.forward(input)
        #     print(input)
        #     print(input.shape)
        return input

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# plt.imshow(train_images[0])
# plt.show()

nn = Network([
    Flattern(),
    Dense(16),
    Activation(Functions.RELU),
    Dense(16),
    Activation(Functions.RELU),
    Dense(10),
    Activation(Functions.SIGMOID)
    ])

nn.compile([28 * 28, 1])
# start = time.time()
# for i in range(train_images.shape[0]):
print(nn.forward(train_images[0:5] / 255))
# print(time.time() - start)

