import numpy as np
from layers import Layer
from typing import List
from loss import mean_squared_err, mean_squared_err_deriv
from tqdm import tqdm


class Network:
    def __init__(self, layers: List[Layer] = []) -> None:
        self.layers = layers

    def get_loss(self, input, expected):
        results = self.forward(input)
        return mean_squared_err(expected, results)

    def get_accuracy(self, input, expected):
        results = self.forward(input)
        predictions = results.argmax(axis=1)
        actual = expected.argmax(axis=1)
        return len(np.where(predictions == actual)[0]) / input.shape[0]

    def forward(self, input: np.ndarray) -> np.ndarray:
        [input := layer.forward(input) for layer in self.layers]
        return input

    def fit(
        self,
        epocs: int,
        batchsize: int,
        learning_rate: float,
        train_input: np.ndarray,
        train_output: np.ndarray,
        test_input: np.ndarray,
        test_output: np.ndarray,
    ):
        for i in range(epocs):
            print(f"Epoch: {i + 1}/{epocs}")
            self.epoch(batchsize, learning_rate, train_input, train_output)
            print(f"Loss: {self.get_loss(test_input, test_output)}")
            print(f"Accuracy: {self.get_accuracy(test_input, test_output)}")

    def epoch(self, batchsize, learning_rate, input: np.ndarray, expected: np.ndarray):
        p = np.random.permutation(input.shape[0])
        input, expected = input[p], expected[p]
        for i in tqdm(
            range(0, input.shape[0], batchsize), unit=" batches", ascii=" >>>>>>>>>>="
        ):
            res = self.forward(input[i : i + batchsize])
            cost = mean_squared_err_deriv(expected[i : i + batchsize], res)
            [
                cost := layer.backward(cost, learning_rate)
                for layer in reversed(self.layers)
            ]
