import numpy as np
import yfinance as yf
import pandas as pd
import statsmodels.api as sm

def add_stocks_from_tickers(tickers, **kwargs):
    """ Takes a stock ticker string, calculates the returns, and inserts them into a data frame

    #fetching-data-for-multiple-tickers to see how they work.
    Takes optional period and interval keyword args. Go to https://github.com/ranaroussi/yfinance
    """

    close_prices = yf.download(tickers, period=kwargs.get('period', "5y"), interval=kwargs.get(
        'interval', "1d"))[['Close']].dropna()
      
    return close_prices.apply(lambda ticker: get_returns(ticker))

def get_returns(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Takes a datafrom of closing prices, calculates the returns, and returns them as a dataframe"""
    offset_close = close_prices.shift(-1)
    offset_returns = offset_close / close_prices - 1
    return offset_returns.shift(1).dropna()


def add_factors_from_tickers(factors, **kwargs):
    """Takes a factor name (for better printing), a ticker representing a factor (such as a thematic index ticker) and the same kwargs as the constructor"""
    return add_stocks_from_tickers(factors, **kwargs)

def regress_factors(stock_df: pd.Series, factors_df: pd.DataFrame):
    """Takes a list of factors that you want to regress on and will print a summary of a multiple regression on those factors with the objects stock

    Can pass in self.factor_names to regress on all factors
    """
    SIGNIF_LEVEL = 0.05

    stock = stock_df.name

    portfolios = []
    ticker = stock[1] if isinstance(stock, tuple) else stock
    model = sm.OLS(stock_df, factors_df)
    results = model.fit()
    print(results.summary())
    factor_pvals = zip(results.pvalues.index, results.pvalues.values)
    for factor_pval in factor_pvals:
      factor, pvalue = factor_pval
      if pvalue < SIGNIF_LEVEL:
          result = [str(factor), [ticker]]
          portfolios.append(result)
      
    return portfolios

def debug_shape(dfs):
  for df in dfs:
    print(df.shape)

def add_factors_from_csv(file) -> pd.DataFrame:
    """Takes a factor name and a CSV file, calculates the returns and returns them as a dataframe

    The first two elements should be a date column heading and a "Close" (case sensitive) column heading. The data should be two columns corresponding to dates and closing prices. 
    This method is untested and may need to be modified.
    """
    factor_close = pd.read_csv(file, index_col=0)[['close']]
    return get_returns(factor_close)

kwargs = {'interval':'1mo'}
stock_returns = add_stocks_from_tickers(['AAPL', 'PLD', 'NFLX'])
factors = add_factors_from_tickers(['^GSPC', '^DJI'])

# sml = add_factors_from_csv('./SML.csv')

debug_shape([stock_returns, factors])

stock_return = []
for stock in stock_returns:
    stock_return.append(stock_returns[stock])

stocks_rdd = sc.parallelize(stock_return)
stocks_rdd = stocks_rdd.flatMap(lambda j: regress_factors(j, factors))
stocks_rdd = stocks_rdd.reduceByKey(lambda a, b: a+b)
stocks_rdd.collect()

