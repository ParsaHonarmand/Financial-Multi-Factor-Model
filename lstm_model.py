import math

import keras.layers
import tensorflow

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM
import linear_regression as lr
import matplotlib.pyplot as plot

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
    training_expected_targets.append(train_stock_data[i+prediction_range-1:i+prediction_range, 0])

training_input_variables = np.array(training_input_variables)
training_expected_targets = np.array(training_expected_targets)

# Creating the sequential model and run it
model = keras.Sequential([
    keras.layers.Dense(11, input_dim=11, activation=keras.activations.relu, use_bias=True),
    keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True),
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(training_input_variables, training_expected_targets, batch_size=16, epochs=100)




