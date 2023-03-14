from typing import List
import numpy as np
from activation import Activation, Functions
from dense import Dense
from layers import Layer
from flattern import Flattern
from keras.datasets import mnist
from keras.utils import np_utils
from loss import *


class Network:
    def __init__(self, layers: List[Layer] = []) -> None:
        self.layers = layers

    def test(self, input, expected):
        results = self.forward(input)
        cost = mean_squared_err(expected, results)
        print(cost)

    def compile(self, input_shape: List[int]):
        last_shape = input_shape
        for layer in self.layers:
            layer.compile(last_shape)
            last_shape = layer.get_output_shape()

    def forward(self, input: np.ndarray) -> np.ndarray:
        [input := layer.forward(input) for layer in self.layers]
        return input

    def train(self, batchsize, input: np.ndarray, expected: np.ndarray):
        p = np.random.permutation(input.shape[0])
        input, expected = input[p], expected[p]
        for i in range(0, input.shape[0], batchsize):
            res = self.forward(input[i : i + batchsize])
            cost = mean_squared_err_deriv(expected[i : i + batchsize], res)
            # print(cost)
            [cost := layer.backward(cost, 0.1) for layer in reversed(self.layers)]


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

nn = Network(
    [
        Flattern(),
        Dense(16),
        Activation(Functions.SIGMOID),
        Dense(16),
        Activation(Functions.SIGMOID),
        Dense(10),
        Activation(Functions.SIGMOID),
    ]
)

nn.compile([28 * 28, 1])

for _ in range(30):
    nn.test(test_images, test_labels)
    nn.train(10, train_images, train_labels)
