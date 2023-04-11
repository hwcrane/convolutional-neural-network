from activation import Activation
from dense import Dense
from flatten import Flatten
from network import Network
from convolutional import Conv2D

from keras.datasets import mnist
from keras.utils import np_utils


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
train_images = train_images.reshape(*train_images.shape, -1)
test_images = test_images / 255.0
test_images = test_images.reshape(*test_images.shape, -1)
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

nn = Network(
    [
        Conv2D(1, 1, 3),
        Flatten(),
        Activation('relu'),
        Dense(10),
        Activation('sigmoid'),
    ]
)

nn.fit(30, 10, 0.1, train_images, train_labels, test_images, test_labels)
