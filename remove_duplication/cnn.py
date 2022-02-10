# Evaluate individual fitness here
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.utils import to_categorical

import random

INPUT_SHAPE = 28

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train, x_test = x_train.astype("float32"), x_test.astype("float32")
    x_train, x_test = x_train / 255., x_test / 255.

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test

class CNN(object):
    def __init__(self):
        self.x_train, self.y_train, self.x_test, self.y_test = load_data()
        self.rgl_dct = {
            0: regularizers.L1(1e-4),
            1: regularizers.L2(1e-4),
            2: regularizers.L1L2(l1=1e-4, l2=1e-4),
            3: None
        }

    def choose_data(self):
        small_x, small_y = shuffle(self.x_train, self.y_train)
        return small_x[:5000], small_y[:5000]

    def add_dense_layers(self, components, model):
        new_model = model
        if components["nd"] == 1:

            if components["dt"][0] == "feed-forward":
                new_model.add(layers.Flatten())
                new_model.add(
                    layers.Dense(components["dn"][0], 
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0]
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "recurrent":
                last_output = new_model.layers[-1].output_shape
                new_model.add(
                    tf.keras.layers.Reshape((last_output[1] * last_output[2], last_output[3]), 
                    input_shape=last_output
                ))
                new_model.add(layers.SimpleRNN(
                    components["dn"][0], 
                    kernel_regularizer=self.rgl_dct[components["dr"][0]],
                    activation=components["da"][0]
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "GRU":
                last_output = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape((
                    last_output[1] * last_output[2], last_output[3]), 
                    input_shape=last_output
                ))
                new_model.add(layers.GRU(
                    components["dn"][0], 
                    kernel_regularizer=self.rgl_dct[components["dr"][0]],
                    activation=components["da"][0]
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "LSTM":
                new_model.add(layers.TimeDistributed(layers.LSTM(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0]
                )))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.Flatten())
        else:

            if components["dt"][0] == "feed-forward" and components["dt"][1] == "feed-forward":
                new_model.add(layers.Flatten())
                new_model.add(layers.Dense(
                    components["dn"][0], 
                    kernel_regularizer=self.rgl_dct[components["dr"][0]],
                    activation=components["da"][0]))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.Dense(
                    components["dn"][1], 
                    kernel_regularizer=self.rgl_dct[components["dr"][1]],
                    activation=components["da"][1]))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "feed-forward" and components["dt"][1] == "recurrent":
                new_model.add(layers.Flatten())
                new_model.add(layers.Dense(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0]
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape((
                    last_shape[1] // 2, 2), 
                    input_shape=last_shape
                ))
                new_model.add(layers.SimpleRNN(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]],
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "feed-forward" and components["dt"][1] == "GRU":
                new_model.add(layers.Dense(
                    components["dn"][0], 
                    kernel_regularizer=self.rgl_dct[components["dr"][0]],
                    activation=components["da"][0]
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape((
                    last_shape[1] * last_shape[2], last_shape[3]), 
                    input_shape=last_shape
                ))
                new_model.add(layers.GRU(
                    components["dn"][1], 
                    kernel_regularizer=self.rgl_dct[components["dr"][1]],
                    activation=components["da"][0]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "feed-forward" and components["dt"][1] == "LSTM":
                new_model.add(layers.Dense(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0]
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.TimeDistributed(layers.LSTM(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                )))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.Flatten())

            if components["dt"][0] == "recurrent" and components["dt"][1] == "feed-forward":
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] * last_shape[2], last_shape[3]), 
                    input_shape=last_shape
                ))
                new_model.add(layers.SimpleRNN(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0]
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.Dense(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "recurrent" and components["dt"][1] == "recurrent":
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] * last_shape[2], last_shape[3]), 
                    input_shape=last_shape
                ))
                new_model.add(layers.SimpleRNN(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                    return_sequences=True
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.SimpleRNN(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "recurrent" and components["dt"][1] == "GRU":
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] * last_shape[2], last_shape[3]), 
                    input_shape=last_shape
                ))
                new_model.add(layers.SimpleRNN(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                    return_sequences=True
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.GRU(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "recurrent" and components["dt"][1] == "LSTM":
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] * last_shape[2], last_shape[3]), 
                    input_shape=last_shape
                ))
                new_model.add(layers.SimpleRNN(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                    return_sequences=True
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.LSTM(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "GRU" and components["dt"][1] == "feed-forward":
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] * last_shape[2], last_shape[3]), 
                    input_shape=last_shape
                ))
                new_model.add(layers.GRU(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.Dense(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "GRU" and components["dt"][1] == "recurrent":
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] * last_shape[2], last_shape[3]), 
                    input_shape=last_shape
                ))
                new_model.add(layers.GRU(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                    return_sequences=True
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.SimpleRNN(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "GRU" and components["dt"][1] == "GRU":
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] * last_shape[2], last_shape[3]), 
                    input_shape=last_shape
                ))
                new_model.add(layers.GRU(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                    return_sequences=True
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.GRU(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "GRU" and components["dt"][1] == "LSTM":
                last_shape = new_model.layers[-1].output_shape
                new_model.add(tf.keras.layers.Reshape(
                    (last_shape[1] * last_shape[2], last_shape[3]), 
                    input_shape=last_shape
                ))
                new_model.add(layers.GRU(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                    return_sequences=True
                ))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.LSTM(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "LSTM" and components["dt"][1] == "feed-forward":
                new_model.add(layers.TimeDistributed(layers.LSTM(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                )))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.Flatten())
                new_model.add(layers.Dense(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "LSTM" and components["dt"][1] == "recurrent":
                new_model.add(layers.TimeDistributed(layers.LSTM(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                )))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.SimpleRNN(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "LSTM" and components["dt"][1] == "GRU":
                new_model.add(layers.TimeDistributed(layers.LSTM(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                )))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.GRU(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))

            if components["dt"][0] == "LSTM" and components["dt"][1] == "LSTM":
                new_model.add(layers.TimeDistributed(layers.LSTM(
                    components["dn"][0],
                    kernel_regularizer=self.rgl_dct[components["dr"][0]], 
                    activation=components["da"][0],
                )))
                if components["dd"][0] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.LSTM(
                    components["dn"][1],
                    kernel_regularizer=self.rgl_dct[components["dr"][1]], 
                    activation=components["da"][1]
                ))
                if components["dd"][1] == 0.5:
                    new_model.add(layers.Dropout(0.5))
                new_model.add(layers.Flatten())

        new_model.add(layers.Dense(10))

        return new_model

    def build_model(self, components):
        model = models.Sequential()
        model.add(layers.Conv2D(
            components["ck"][0],
            (components["cs"][0], components["cs"][0]),
            activation=components["ca"][0],
            input_shape=(28, 28, 1)
        ))
        model.add(layers.MaxPooling2D((components["cp"][0], components["cp"][0])))

        for i in range(1, components["nc"]):
            model.add(layers.Conv2D(
                components["ck"][i],
                (components["cs"][i], components["cs"][i]),
                activation=components["ca"][i],
            ))
            model.add(layers.MaxPooling2D((components["cp"][i], components["cp"][i])))

        model = self.add_dense_layers(components, model)

        return model

    def is_valid_model(self, conv_sizes, pooling_sizes):
        shape = INPUT_SHAPE
        for i in range(len(conv_sizes)):
            if shape - conv_sizes[i] + 1 <= 0:
                return False
            else:
                shape = shape - conv_sizes[i] + 1
            if shape - pooling_sizes[i] < 0:
                return False
            else:
                shape = (shape - pooling_sizes[i]) // pooling_sizes[i] + 1
        return True

    def get_optimizers(self, function, lr):
        learning_rate = lr
        if function == 0:
            opt = optimizers.SGD(learning_rate=learning_rate)
        elif function == 1:
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.1)
        elif function == 2:
            opt = optimizers.SGD(learning_rate=learning_rate, nesterov=True)
        elif function == 3:
            opt = optimizers.Adagrad(learning_rate=learning_rate)
        elif function == 4:
            opt = optimizers.Adamax(learning_rate=learning_rate)
        elif function == 5:
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif function == 6:
            opt = optimizers.Adadelta(learning_rate=learning_rate)
        elif function == 7:
            opt = optimizers.RMSprop(learning_rate=learning_rate)

        return opt

    def evaluate(self, components):
        if not self.is_valid_model(components["cs"], components["cp"]):
            return 0
        else:
            # return random.randint(1, 10000000)
            model = self.build_model(components)
            opt = self.get_optimizers(components["f"], components["n"])
            model.compile(optimizer=opt,
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
            small_x, small_y = self.choose_data()
            batch_size = components["b"]
            model.fit(small_x, small_y, epochs=5, validation_data=(self.x_test, self.y_test), batch_size=batch_size, verbose=0)
            _, test_acc = model.evaluate(self.x_test,  self.y_test, verbose = 0)
            del model
            return test_acc


