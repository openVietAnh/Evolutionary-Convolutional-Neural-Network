import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets, layers, models, regularizers, optimizers, initializers
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

model = models.Sequential()
model1 = models.Sequential()
model2 = models.Sequential()
model3 = models.Sequential()
model4 = models.Sequential()
model5 = models.Sequential()
model6 = models.Sequential()
model7 = models.Sequential()
model8 = models.Sequential()
model9 = models.Sequential()
model10 = models.Sequential()
model11 = models.Sequential()
model12 = models.Sequential()
model13 = models.Sequential()
model14 = models.Sequential()
model15 = models.Sequential()

model._name = "Dense to dense"
model1._name = "Dense to recurrent"
model2._name = "Dense to LSTM"
model3._name = "Dense to GRU"
model4._name = "LSTM to LSTM"
model5._name = "LSTM to dense"
model6._name = "LSTM to recurrent"
model7._name = "LSTM to GRU"
model8._name = "GRU to GRU"
model9._name = "GRU to LSTM"
model10._name = "GRU to dense"
model11._name = "GRU to recurrent"
model12._name = "recurrent to recurrent"
model13._name = "recurrent to GRU"
model14._name = "recurrent to dense"
model15._name = "recurrent to LSTM"

# Convolutional layers
model.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((3, 3)))

model1.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model1.add(layers.MaxPooling2D((3, 3)))
model1.add(layers.Conv2D(128, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((3, 3)))

model2.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model2.add(layers.MaxPooling2D((3, 3)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((3, 3)))

model3.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model3.add(layers.MaxPooling2D((3, 3)))
model3.add(layers.Conv2D(128, (3, 3), activation='relu'))
model3.add(layers.MaxPooling2D((3, 3)))

model4.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model4.add(layers.MaxPooling2D((3, 3)))
model4.add(layers.Conv2D(128, (3, 3), activation='relu'))
model4.add(layers.MaxPooling2D((3, 3)))

model5.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model5.add(layers.MaxPooling2D((3, 3)))
model5.add(layers.Conv2D(128, (3, 3), activation='relu'))
model5.add(layers.MaxPooling2D((3, 3)))

model6.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model6.add(layers.MaxPooling2D((3, 3)))
model6.add(layers.Conv2D(128, (3, 3), activation='relu'))
model6.add(layers.MaxPooling2D((3, 3)))

model7.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model7.add(layers.MaxPooling2D((3, 3)))
model7.add(layers.Conv2D(128, (3, 3), activation='relu'))
model7.add(layers.MaxPooling2D((3, 3)))

model8.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model8.add(layers.MaxPooling2D((3, 3)))
model8.add(layers.Conv2D(128, (3, 3), activation='relu'))
model8.add(layers.MaxPooling2D((3, 3)))

model9.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model9.add(layers.MaxPooling2D((3, 3)))
model9.add(layers.Conv2D(128, (3, 3), activation='relu'))
model9.add(layers.MaxPooling2D((3, 3)))

model10.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model10.add(layers.MaxPooling2D((3, 3)))
model10.add(layers.Conv2D(128, (3, 3), activation='relu'))
model10.add(layers.MaxPooling2D((3, 3)))

model11.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model11.add(layers.MaxPooling2D((3, 3)))
model11.add(layers.Conv2D(128, (3, 3), activation='relu'))
model11.add(layers.MaxPooling2D((3, 3)))

model12.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model12.add(layers.MaxPooling2D((3, 3)))
model12.add(layers.Conv2D(128, (3, 3), activation='relu'))
model12.add(layers.MaxPooling2D((3, 3)))

model13.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model13.add(layers.MaxPooling2D((3, 3)))
model13.add(layers.Conv2D(128, (3, 3), activation='relu'))
model13.add(layers.MaxPooling2D((3, 3)))

model14.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model14.add(layers.MaxPooling2D((3, 3)))
model14.add(layers.Conv2D(128, (3, 3), activation='relu'))
model14.add(layers.MaxPooling2D((3, 3)))

model15.add(layers.Conv2D(64, (3, 3), activation='linear', input_shape=(28, 28, 1)))
model15.add(layers.MaxPooling2D((3, 3)))
model15.add(layers.Conv2D(128, (3, 3), activation='relu'))
model15.add(layers.MaxPooling2D((3, 3)))

# Dense layers

# 1. Feed-forward -> feed-forward
model.add(layers.Dense(120, activation='linear'))
model.add(layers.Dense(60, activation='relu'))

# 2. Feed-forward -> Recurrent
model1.add(layers.Flatten())
model1.add(layers.Dense(120, activation='linear'))
los = model1.layers[-1].output_shape
model1.add(tf.keras.layers.Reshape((los[1] // 2, 2), input_shape=los))
model1.add(layers.SimpleRNN(128))
model1.add(layers.Dropout(0.2))

# 3. Feed-forward -> LSTM
model2.add(layers.Dense(120, activation='linear'))
model2.add(layers.TimeDistributed(layers.LSTM(128, activation='relu')))
model2.add(layers.Flatten())

# 4. Feed-forward -> GRU
model3.add(layers.Dense(120, activation='linear'))
los = model3.layers[-1].output_shape
model3.add(tf.keras.layers.Reshape((los[1] * los[2], los[3]), input_shape=los))
model3.add(layers.GRU(128, kernel_regularizer='l1'))

# 5. LSTM -> LSTM
model4.add(layers.TimeDistributed(layers.LSTM(128, activation='relu')))
model4.add(layers.LSTM(128, activation='relu'))
model4.add(layers.Flatten())

# 6. LSTM -> feed-forward
model5.add(layers.TimeDistributed(layers.LSTM(128, activation='relu')))
model5.add(layers.Flatten())
model5.add(layers.Dense(60, activation='relu'))

# 7. LSTM -> Recurrent
model6.add(layers.TimeDistributed(layers.LSTM(128, activation='relu')))
model6.add(layers.SimpleRNN(128))

# 8. LSTM -> GRU
model7.add(layers.TimeDistributed(layers.LSTM(128, activation='relu')))
model7.add(layers.GRU(128))

# 9. GRU -> GRU
los = model8.layers[-1].output_shape
model8.add(tf.keras.layers.Reshape((los[1] * los[2], los[3]), input_shape=los))
model8.add(layers.GRU(64, kernel_regularizer='l1', return_sequences=True))
model8.add(layers.GRU(128, kernel_regularizer='l1'))

# 10. GRU -> LSTM
los = model9.layers[-1].output_shape
model9.add(tf.keras.layers.Reshape((los[1] * los[2], los[3]), input_shape=los))
model9.add(layers.GRU(64, kernel_regularizer='l1', return_sequences=True))
model9.add(layers.LSTM(128, kernel_regularizer='l1', activation='relu'))

# 11. GRU -> feed-forward
los = model10.layers[-1].output_shape
model10.add(tf.keras.layers.Reshape((los[1] * los[2], los[3]), input_shape=los))
model10.add(layers.GRU(64))
model10.add(layers.Dense(60, activation='relu'))

# 12. GRU -> Recurrent
los = model11.layers[-1].output_shape
model11.add(tf.keras.layers.Reshape((los[1] * los[2], los[3]), input_shape=los))
model11.add(layers.GRU(64, return_sequences=True))
model11.add(layers.SimpleRNN(128))

# 13. Recurrent -> Recurrent 
los = model12.layers[-1].output_shape
model12.add(tf.keras.layers.Reshape((los[1] * los[2], los[3]), input_shape=los))
model12.add(layers.SimpleRNN(128, kernel_regularizer='l1', return_sequences=True))
model12.add(layers.SimpleRNN(60, kernel_regularizer='l1'))

# 14. Recurrent -> GRU
los = model13.layers[-1].output_shape
model13.add(tf.keras.layers.Reshape((los[1] * los[2], los[3]), input_shape=los))
model13.add(layers.SimpleRNN(128, kernel_regularizer='l1', return_sequences=True))
model13.add(layers.GRU(64, kernel_regularizer='l1'))

# 15. Recurrent -> feed-forward
los = model14.layers[-1].output_shape
model14.add(tf.keras.layers.Reshape((los[1] * los[2], los[3]), input_shape=los))
model14.add(layers.SimpleRNN(128, kernel_regularizer='l1'))
model14.add(layers.Dense(60))

# 16. Recurrent -> LSTM
los = model15.layers[-1].output_shape
model15.add(tf.keras.layers.Reshape((los[1] * los[2], los[3]), input_shape=los))
model15.add(layers.SimpleRNN(128, return_sequences=True))
model15.add(layers.LSTM(128, activation='relu'))

# Final output layer
model.add(layers.Dense(10))
model1.add(layers.Dense(10))
model2.add(layers.Dense(10))
model3.add(layers.Dense(10))
model4.add(layers.Dense(10))
model5.add(layers.Dense(10))
model6.add(layers.Dense(10))
model7.add(layers.Dense(10))
model8.add(layers.Dense(10))
model9.add(layers.Dense(10))
model10.add(layers.Dense(10))
model11.add(layers.Dense(10))
model12.add(layers.Dense(10))
model13.add(layers.Dense(10))
model14.add(layers.Dense(10))
model15.add(layers.Dense(10))

model.summary(line_length = 100)
model1.summary(line_length = 100)
model2.summary(line_length = 100)
model3.summary(line_length = 100)
model4.summary(line_length = 100)
model5.summary(line_length = 100)
model6.summary(line_length = 100)
model7.summary(line_length = 100)
model8.summary(line_length = 100)
model9.summary(line_length = 100)
model10.summary(line_length = 100)
model11.summary(line_length = 100)
model12.summary(line_length = 100)
model13.summary(line_length = 100)
model14.summary(line_length = 100)
model15.summary(line_length = 100)

# model.compile(
#     optimizer=optimizers.Adagrad(learning_rate=0.05),
#     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=15)
# _, test_acc = model.evaluate(x_test,  y_test, verbose = 2)
# print("Result:", test_acc)
