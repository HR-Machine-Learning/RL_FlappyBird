from new_deepQ_flappy_agent import NewDeepQAgent
from typing import Dict, Tuple
import numpy as np
from sklearn.neural_network import MLPRegressor



model: MLPRegressor = MLPRegressor(hidden_layer_sizes=(100, 10),
                                    activation='logistic',
                                    learning_rate_init=0.1)

# X = np.ndarray(shape=(4,4), dtype=int)
# y = np.ndarray(shape=(4), dtype=float)

# print(X)

X = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
y = np.array([0, 0, 0, 0])

print(X)
print(y)

model.fit(X, y)

print("The prediction is: ", model.predict(X))