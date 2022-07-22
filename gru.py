import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
import math

# This code was taken and partially modified from https://www.kaggle.com/rodsaldanha/stock-prediction-pytorch

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def split_data(stock, batch_size):
    data_raw = stock
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - batch_size):
        data.append(data_raw[index: index + batch_size])

    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return (train_set_size, [x_train, x_test, y_train, y_test])

def convert_to_tensor(np_arr):
    return torch.from_numpy(np_arr).type(torch.Tensor)


def denormalize(tensor, name): 
    return pd.DataFrame(scaler.inverse_transform(tensor.detach().numpy()), columns=[name]) # Convert normalized data back into prices

def results_to_csv(pred, actual, rmse):
    pred_and_actual = pd.concat([pred, actual], axis=1, join='outer')
    file_name = f'./NN_results/{ticker}_NN_results.xlsx'
    pred_and_actual.to_excel(file_name, sheet_name=f'{ticker}')

    with pd.ExcelWriter(file_name, mode = 'a') as writer:
        rmse = pd.Series([rmse], name='RMSE')
        rmse.to_excel(writer, sheet_name=f'{ticker} RMSE')


tickers = ['F', 'AMZN', 'COST', 'TFC']

period = {
    'start': datetime.datetime(2014, 1, 3),
    'end': datetime.datetime(2021, 12, 1)
}

for ticker in tickers:
    prices = yf.download(ticker, **period)[['Close']].dropna()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_prices = scaler.fit_transform(prices['Close'].values.reshape(-1,1))

    test_start_date, data = split_data(normalized_prices, batch_size=25)

    starting_date = str(prices.iloc[test_start_date:].index[0])[:10]

    x_train, x_test, y_train_gru, y_test_gru = [*map(convert_to_tensor, data)]

    model = GRU(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    start_time = time.time()

    num_epochs = 60
    for epoch in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train_gru)
        print("Epoch ", epoch, "MSE: ", loss.item())

        optimiser.zero_grad() # Zero the tensor gradients
        loss.backward() # Calculate the gradient used in gradient descent
        optimiser.step() # Take one step in the opposite direction of the gradient

    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))

    y_test_pred = model(x_test)

    denormalized_pred = denormalize(y_test_pred, 'Predicted')
    denormalized_test = denormalize(y_test_gru, 'Actual')


    print('\n################## STATISTICS ##################')
    score = math.sqrt(mean_squared_error(denormalized_pred, denormalized_test))
    print(f'Root Mean Squared Error (RMSE) of {ticker} Neural Net: ${score:.2f}')

    price_on_last_day_df: pd.Series = prices.loc[period['end'] - datetime.timedelta(days=1)]
    price_on_last_day = price_on_last_day_df.values[0]

    print(f'\n{ticker} price as of {starting_date}: ${price_on_last_day:.2f}')
    print(f'RMSE relative to price as of {str(period["end"])[:10]}: {score:.2f}/{price_on_last_day:.2f} = {score/price_on_last_day*100:.2f}%\n')

    plt.plot(denormalized_pred, label="Predicted Price")
    plt.plot(denormalized_test, label="Actual Price")
    plt.ylabel(f'Price of {ticker}')
    plt.xlabel(f'Days since {starting_date}')
    plt.title(ticker)
    plt.legend()
    plt.savefig(f'./NN_results/{ticker}_GRU_Pred.png')

    results_to_csv(denormalized_pred, denormalized_test, score)

    # Clean Up
    plt.clf()
    del denormalized_pred
    del denormalized_test
    del model