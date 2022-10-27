import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV



x = 0 # normalized feature of the state, normalized between [-1, 1]
y = 0 # the two actions
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)


sc=StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)


mlp_reg = MLPRegressor(hidden_layer_sizes=(100,10), # 100 neurons in the first layer, 10 neurons in the second layer
                       activation = 'logistic', # sigmoid activation function for all hidden layers
                       solver = 'adam', 
                       learning_rate_init = 0.01) # initial learning rate of 0.01

mlp_reg.fit(trainX_scaled, trainY)