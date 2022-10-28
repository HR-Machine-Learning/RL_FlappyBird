import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


class NeuralNetwork:
    def __init__(self, hidden_layers, activation_function, initial_learning_rate):
        # list of hidden layers and the number of neurons in each layer
        self.hidden_layers = hidden_layers
        self.activation_function = activation_function
        self.initial_learning_rate = initial_learning_rate
        self.y = [0, 1]
        # I think the output layer will be infer from the y feature and doesn't need to be specified
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layers,
                                  activation=self.activation_function,
                                  learning_rate_init=self.initial_learning_rate)

    # Every 1000 frames, we shuffle the date and we train the network using batches of choosen size
    def partial_training(self, X_train: np.array, batch_size: int = 100):
        X_train = X_train.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(X_train), batch_size):
            self.model.partial_fit(X_train[i:i+batch_size], self.y)

    # Normalize the input features of the state to the interval [-1, 1]
    def normalize(self, X):
        # TODO: check if this is the correct normalization
        X_normalized = preprocessing.normalize(X, norm='l2')
        return X_normalized

    def predict_(self, state):
        return 