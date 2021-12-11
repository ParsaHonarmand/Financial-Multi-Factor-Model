# Import modules and packages
import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import linear_regression as lr
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# Read in the data
training_data = pd.read_csv('stock_and_factor_prices.csv')
cols = list(training_data)[2:len(training_data.columns)]

# Cleaning up the data
training_dates = list(training_data['Date'])
training_dates = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in training_dates]

training_data = training_data[cols].astype(str)
for i in cols:
    for j in range(0, len(training_data)):
        training_data[i][j] = training_data[i][j].replace(',', '')
training_data = training_data.astype(float)
training_set = training_data.to_numpy()

# Scale the data
scaler = StandardScaler()
training_set_scaled = scaler.fit_transform(training_set)

prediction_scaler = StandardScaler()
predictions = prediction_scaler.fit_transform(training_set[:, 0:1])

# Create training inputs
x_train = []
y_train = []
prediction_days = 20
historical_days = 200

# (60000, 28, 28)
# (60000,)
# (10000, 28, 28)
# (10000,)

for i in range(historical_days, len(training_set_scaled) - prediction_days + 1):
    x_train.append(training_set_scaled[i - historical_days:i, 0:training_data.shape[1] - 1])
    y_train.append(training_set_scaled[i + prediction_days - 1][0])

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)

# Create the model
inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
dense = layers.Dense(x_train.shape[1]*x_train.shape[2], activation="relu")
x = dense(inputs)
x = layers.Dense(200, activation="relu")(x)
outputs = layers.Dense(prediction_days)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

# model.fit(x_train, y_train, epochs=2, validation_split=0.2)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)