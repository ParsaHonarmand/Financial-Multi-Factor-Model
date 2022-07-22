import math

import keras.layers
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers

import linear_regression as lr
import numpy as np

plot.style.use('fivethirtyeight')

stock = 'MSFT'
# Get stock data
lr.add_stocks_from_tickers([stock])
stock_data = pd.read_csv(lr.PRICE_DOWNLOADS_CSV)

# Get factor data
lr.add_factors_from_csv(lr.FACTOR_DIRECTORY)
factor_data = pd.read_csv(lr.PRICE_FACTOR_CSV)

# Match rows between stock and factors
combined_data = lr.normalizeFactorDates(stock_data, factor_data)
combined_data = combined_data.set_index('Date')
# combined_data.to_csv("stock_and_factor_prices.csv")

# Isolate stock close data and factor close data
stock_data = combined_data.filter([stock]).values
training_data_length = math.ceil(len(stock_data) * 0.75)

factor_names = []
for i in range(1, len(combined_data.columns)):
    factor_names.append(combined_data.columns[i])

factor_data = combined_data.filter(factor_names).values

# Transform and scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_stock_data = scaler.fit_transform(stock_data)
scaled_factor_data = scaler.fit_transform(factor_data)

scaled_stock_data = np.array(scaled_stock_data)
scaled_factor_data = np.array(scaled_factor_data)

# Create training data and set training hyper parameters
prediction_range = 20
historical_range = 200
train_stock_data = scaled_stock_data[0:training_data_length, :]
train_factor_data = scaled_factor_data[0:training_data_length, :]

training_input_variables = []
training_expected_targets = []
for i in range(historical_range, len(train_stock_data) - prediction_range + 1):
    training_input_variables.append(train_factor_data[(i - historical_range):i])
    training_expected_targets.append(train_stock_data[i + prediction_range - 1][0])

training_input_variables = np.array(training_input_variables)
training_expected_targets = np.array(training_expected_targets)
# (1102, 200, 11)
# (1102,)
training_input_variables = training_input_variables.reshape(-1, historical_range * len(factor_names))
# (1102, 2200)

# Create the functional model
inputs = keras.Input(shape=(training_input_variables.shape[1],))
dense = layers.Dense(training_input_variables.shape[1], activation="relu")
x = dense(inputs)
x = layers.Dense(200, activation="relu")(x)
outputs = layers.Dense(prediction_range)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

model.fit(training_input_variables, training_expected_targets, batch_size=training_input_variables.shape[1], epochs=2,
          validation_split=0.2)

# Create test data set
test_stock_data = scaled_stock_data[len(train_stock_data):, :]
test_factor_data = scaled_factor_data[len(train_factor_data):, :]

test_stock_data = np.array(test_stock_data)
test_factor_data = np.array(test_factor_data)

test_input_variables = []
test_expected_targets = []
for i in range(historical_range, len(test_stock_data) - prediction_range + 1):
    test_input_variables.append(test_factor_data[(i - historical_range):i])
    test_expected_targets.append(test_stock_data[i + prediction_range - 1:i + prediction_range, 0])

test_input_variables = np.array(test_input_variables)
test_expected_targets = np.array(test_expected_targets)
test_input_variables = test_input_variables.reshape(-1, historical_range * len(factor_names))
test_scores = model.evaluate(test_input_variables, test_expected_targets, verbose=2)

print(test_scores)
