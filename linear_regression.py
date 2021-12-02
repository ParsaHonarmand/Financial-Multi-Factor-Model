import numpy as np
from pandas.core.arrays.categorical import factorize_from_iterable
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import chart_studio.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import statsmodels.api as sm
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

class FactorReg: 
  def __init__(self, ticker, **kwargs):
    self.stock = yf.download(ticker, period=kwargs.get('period', "5y"), interval=kwargs.get('interval', "1d"))[['Close']].dropna().rename(columns = {'Close': ticker}) 
    self.factors = pd.DataFrame()

  def calc_beta(self, factor):
    stock_np = self.stock.to_numpy().flatten()
    factor_np = self.factors[factor].to_numpy()

    # print(stock_np)
    # print(stock_np.shape)
    # print(factor_np.shape)
    # exit()

    cov = np.cov(stock_np, factor_np)
    var = np.var(factor_np)

    # print(cov)
    # print(var)
    print(cov/var)



  def add_factor_from_ticker(self, factor_name, ticker, **kwargs):
    self.factors[factor_name] = yf.download(ticker, period=kwargs.get('period', "5y"), interval=kwargs.get('interval', "1d"), group_by='ticker')[['Close']].dropna()  

  def add_factor_from_csv(self, factor_name, file):
    self.factors[factor_name] = pd.read_csv(file, index_col=0)[['Close']]

  def regress_factor(self, regress_factors):
    model = LinearRegression()
    df = self.factors[[regress_factors]]
    model.fit(df, self.stock)

    print("Model Coefficients:", model.coef_)
    # print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    # print("Coefficient of Determination:", r2_score(y_test, y_pred))

  def debug(self):
    print(self.stock.head())
    print(self.factors.head())

factor_reg = FactorReg('PLD', interval='1mo')
factor_reg.add_factor_from_ticker('SPY', 'SPY', interval='1mo')
factor_reg.calc_beta('SPY')
# factor_reg.debug()
factor_reg.regress_factor('SPY')

