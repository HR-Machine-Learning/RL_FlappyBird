from sklearn.neural_network import MLPRegressor¶
from sklearn.model_selection import train_test_split¶

class NN:
    
    def __init__(self, hidden_layers, activation_function, initial_learning_rate):
        self.hidden_layers = hidden_layers # list of hidden layers and the number of neurons in each layer
        self.activation_function = activation_function 
        self.initial_learning_rate = initial_learning_rate
        # I think the output layer will be infer from the y feature and doesn't need to be specified    
        self.model = MLPRegressor(hidden_layer_sizes = self.hidden_layers, 
                                  activation = self.activation_function, 
                                  learning_rate_init = self.initial_learning_rate)