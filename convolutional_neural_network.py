# Evaluate individual fitness here
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.keras.utils import to_categorical

import random

INPUT_SHAPE = 28

class CNN(object):
    def __init__(self):
        pass

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

    def evaluate(self, components):
        if not self.is_valid_model(components["cs"], components["cp"]):
            return 0
        else:
            return random.randint(1, 100)


