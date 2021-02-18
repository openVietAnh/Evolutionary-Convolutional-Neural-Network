import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.keras.utils import to_categorical

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train, x_test = x_train.astype("float32"), x_test.astype("float32")
    x_train, x_test = x_train / 255., x_test / 255.

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='linear'))
model.add(layers.Dropout(0.5))

model.summary()

model.compile(
    optimizer=optimizers.Adagrad(learning_rate=0.05),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=15)
_, test_acc = model.evaluate(x_test,  y_test, verbose = 2)
print("Result:", test_acc)
