import os
import struct
import numpy as np

import keras
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

mnist_dir = '/Users/matt/dev/machine-learning/datasets/mnist/'

path_test_labels = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte')
path_test_images = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
path_train_labels = os.path.join(mnist_dir, 'train-labels-idx1-ubyte')
path_train_images = os.path.join(mnist_dir, 'train-images-idx3-ubyte')

class Mnist:
    def __init__(self, labels_path, images_path):
        with open(labels_path, 'rb') as file:
            magic, num = struct.unpack(">II", file.read(8))
            assert magic == 2049
            self.labels = np.fromfile(file, dtype=np.int8)

        with open(images_path, 'rb') as file:
            magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
            assert magic == 2051
            raw_images = np.fromfile(file, dtype=np.int8)
            self.images = raw_images.reshape(num, rows, cols) # Nx28x28

    def flattened_images(self):
        (num_images, rows, cols) = self.images.shape
        return self.images.reshape(num_images, rows*cols) # Nx784

    def one_hot_labels(self):
        return np_utils.to_categorical(self.labels, 10)

train = Mnist(path_train_labels, path_train_images)
test = Mnist(path_test_labels, path_test_images)

batch_size = 128
nb_epoch = 20

# for i in range(0,20):

model = Sequential()
model.add(Dense(512, input_dim=(784)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# model.summary()

model.compile(loss='categorical_crossentropy',
            # optimizer=keras.optimizers.RMSprop(),
            optimizer=keras.optimizers.RMSprop(),
            metrics=['accuracy'])

history = model.fit(train.flattened_images(),
                  train.one_hot_labels(),
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  verbose=1,
                  validation_data=(test.flattened_images(),
                                  test.one_hot_labels()))
score = model.evaluate(test.flattened_images(),
                    test.one_hot_labels(),
                    verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

