from abc import ABC, abstractmethod
import numpy as np
from typing import List


class Layer(ABC):
    def __init__(self) -> None:
        self.initalised = False
        pass

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        pass

    @abstractmethod
    def get_output_shape(self) -> List[int]:
        pass

    @abstractmethod
    def compile(self, input_shape: List[int]):
        pass
