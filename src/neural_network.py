import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from typing import List, Tuple

from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor


class NeuralNetwork:
    def __init__(self, hidden_layers: int, activation_function: str, initial_learning_rate: float) -> None:
        # list of hidden layers and the number of neurons in each layer
        self.hidden_layers: int = hidden_layers
        self.activation_function: str = activation_function
        self.initial_learning_rate: float = initial_learning_rate
        self.y: List[int] = np.zeros(1000, dtype=int)

        self.model: MLPRegressor = MLPRegressor(hidden_layer_sizes=self.hidden_layers,
                                  activation=self.activation_function,
                                  learning_rate_init=self.initial_learning_rate)
        self.initialize()
        

    # Every 1000 frames, we shuffle the date and we train the network using batches of choosen size
    def partial_training(self, X_train: np.ndarray, batch_size: int = 100) -> None:
        X_train = X_train.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(X_train), batch_size):
            self.model.partial_fit(X_train[i:i+batch_size], self.y)

    # Normalize the input features of the state to the interval [-1, 1]
    def normalize(self, X):
        # TODO: check if this is the correct normalization
        X_normalized = preprocessing.normalize(X, norm='l2')
        return X_normalized

    def predict_next_state(self, state: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        gaga = np.array(state).reshape(1, -1) # New shape looks like [[ 7.  5. 15. -7.]]
        return self.model.predict(gaga)
 
    def initialize(self) -> None:
        array: np.ndarray = np.zeros((1000,4), dtype=int)
        print(array)

        #for element in array:
        #    element = [random.randint(0, 15), random.randint(0, 15), random.randint(0, 15), random.randint(-8, 10)]
        #    np.append(array, element)
        
        # print(array)

        self.model.fit(array, self.y)
