from activation import Activation
from dense import Dense
from flattern import Flatten
from keras.datasets import mnist
from keras.utils import np_utils
from network import Network


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

nn = Network(
    [
        Flatten([28, 28]),
        Dense(28 * 28, 16),
        Activation('relu'),
        Dense(16, 16),
        Activation('sigmoid'),
        Dense(16, 10),
        Activation('sigmoid'),
    ]
)

nn.fit(100, 10, 0.1, train_images, train_labels, test_images, test_labels)
