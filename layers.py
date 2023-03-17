from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, derivatives: np.ndarray, learning_rate) -> np.ndarray:
        pass
