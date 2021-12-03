import numpy as np
import yfinance as yf
import pandas as pd
import statsmodels.api as sm

class FactorReg: 

  def __init__(self, ticker, **kwargs):
    """ Takes a stock ticker string, calculates the returns, and inserts them into a data frame 
    
    Takes optional period and interval keyword args. Go to https://github.com/ranaroussi/yfinance#fetching-data-for-multiple-tickers to see how they work. 
    """

    close_prices = yf.download(ticker, period=kwargs.get('period', "5y"), interval=kwargs.get('interval', "1d"))[['Close']].dropna().rename(columns = {'Close': ticker}) 
    self.returns = self.get_returns(close_prices)
    self.factors = pd.DataFrame()
    self.factor_names = []

  def get_returns(self, close_prices):
    """Takes a datafrom of closing prices, calculates the returns, and returns them as a dataframe"""
    offset_close = close_prices.shift(-1)
    offset_returns = offset_close / close_prices - 1
    return offset_returns.shift(1).dropna()

  def add_factor_from_ticker(self, factor_name, ticker, **kwargs):
    """Takes a factor name (for better printing), a ticker representing a factor (such as a thematic index ticker) and the same kwargs as the constructor"""
    factor_close = yf.download(ticker, period=kwargs.get('period', "5y"), interval=kwargs.get('interval', "1d"), group_by='ticker')[['Close']].dropna()  
    factor_returns = self.get_returns(factor_close)
    self.factor_names.append(factor_name)
    self.factors[factor_name] = factor_returns

  def add_factor_from_csv(self, factor_name, file):
    """Takes a factor name and a CSV file, calculates the returns and returns them as a dataframe
    
    The first two elements should be a date column heading and a "Close" (case sensitive) column heading. The data should be two columns corresponding to dates and closing prices. 
    This method is untested and may need to be modified.
    """
    factor_close = pd.read_csv(file, index_col=0)[['Close']]
    factor_returns = self.get_returns(factor_close)
    self.factor_names.append(factor_name)
    self.factors[factor_name] = factor_returns

  def regress_factor(self, regress_factors):
    """Takes a list of factors that you want to regress on and will print a summary of a multiple regression on those factors with the objects stock
    
    Can pass in self.factor_names to regress on all factors
    """
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

