import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


class NN:
    def __init__(self, hidden_layers, activation_function, initial_learning_rate):
        self.hidden_layers = hidden_layers # list of hidden layers and the number of neurons in each layer
        self.activation_function = activation_function 
        self.initial_learning_rate = initial_learning_rate
        # I think the output layer will be infer from the y feature and doesn't need to be specified    
        self.model = MLPRegressor(hidden_layer_sizes = self.hidden_layers, 
                                  activation = self.activation_function, 
                                  learning_rate_init = self.initial_learning_rate)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    # Normalize the input features of the state to the interval [-1, 1]
    def normalize_state(self, state):
        scaler = StandardScaler()
        scaler.fit(state)               # not sure if it is doing the proper range for now
        return scaler.transform(state)
