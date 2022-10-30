from keras.datasets import mnist 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

model = Sequential()

model.add(Dense(units = 100, activation ='sigmoid', input_shape = (4,)))
model.add(Dense(units = 10, activation = 'sigmoid'))
model.add(Dense(units = 2, activation = 'sigmoid'))

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])




X_train = 0
Y_train = 0



# Keras for regression
model = Sequential()
    
model.add(Dense(100, input_dim = 4, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

model.compile(loss = 'mean_squared_error', optimizer = 'adam')

estimator = KerasRegressor(build_fn = model, epochs = 30, batch_size = 100, verbose = 1)

history = estimator.fit(X_train,y_train)
