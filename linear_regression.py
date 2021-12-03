import numpy as np
import yfinance as yf
import pandas as pd
import statsmodels.api as sm

class FactorReg: 
  def __init__(self, ticker, **kwargs):
    close_prices = yf.download(ticker, period=kwargs.get('period', "5y"), interval=kwargs.get('interval', "1d"))[['Close']].dropna().rename(columns = {'Close': ticker}) 
    self.returns = self.get_returns(close_prices)
    self.factors = pd.DataFrame()

  def get_returns(self, close_prices):
    offset_close = close_prices.shift(-1)
    offset_returns = offset_close / close_prices - 1
    return offset_returns.shift(1).dropna()

  def add_factor_from_ticker(self, factor_name, ticker, **kwargs):
    factor_close = yf.download(ticker, period=kwargs.get('period', "5y"), interval=kwargs.get('interval', "1d"), group_by='ticker')[['Close']].dropna()  
    factor_returns = self.get_returns(factor_close)
    self.factors[factor_name] = factor_returns

  def add_factor_from_csv(self, factor_name, file):
    factor_close = pd.read_csv(file, index_col=0)[['Close']]
    factor_returns = self.get_returns(factor_close)
    self.factors[factor_name] = factor_returns

  def regress_factor(self, regress_factors):
    df = self.factors[[regress_factors]]
    model = sm.OLS(self.returns, df)
    results = model.fit()
    print(results.summary())

  def debug(self):
    print(self.returns.head())
    print(self.factors.head())

factor_reg = FactorReg('PLD', interval='1mo')
factor_reg.add_factor_from_ticker('SPY', 'SPY', interval='1mo')
factor_reg.regress_factor('SPY')

