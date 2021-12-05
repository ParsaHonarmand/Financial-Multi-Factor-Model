import numpy as np
from pandas.core.algorithms import factorize
from pandas.core.indexing import IndexSlice
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import os


def add_stocks_from_tickers(tickers: list[str], **kwargs: dict[str, str]) -> pd.DataFrame:
    """ Takes a stock ticker string, calculates the returns, and inserts them into a data frame

    #fetching-data-for-multiple-tickers to see how they work.
    Takes optional period and interval keyword args. Go to https://github.com/ranaroussi/yfinance
    """
    # close_prices = yf.download(tickers, period=kwargs.get('period', "10y"), interval=kwargs.get(
    #     'interval', "1d"))[['Close']].dropna()

    close_prices = yf.download(tickers, start="2011-12-01", end="2021-12-01", interval=kwargs.get(
        'interval', "1d"))[['Close']].dropna()

    return close_prices.apply(lambda ticker: get_returns(ticker))


def get_returns(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Takes a datafrom of closing prices, calculates the returns, and returns them as a dataframe"""
    offset_close = close_prices.shift(-1)
    offset_returns = offset_close / close_prices - 1
    return offset_returns.shift(1).dropna()


def add_factors_from_tickers(factors: list[str], **kwargs: dict[str, str]):
    """Takes a factor name (for better printing), a ticker representing a factor (such as a thematic index ticker) and the same kwargs as the constructor"""
    return add_stocks_from_tickers(factors, **kwargs)


def regress_factors(stocks_df: pd.DataFrame, factors_df: pd.DataFrame):
    """Takes a list of factors that you want to regress on and will print a summary of a multiple regression on those factors with the objects stock

    Can pass in self.factor_names to regress on all factors
    """
    SIGNIF_LEVEL = 0.05

    portfolios = dict()
    for stock in stocks_df:
        single_stock_df = stocks_df[stock]

        model = sm.OLS(single_stock_df, factors_df)
        results = model.fit()
        print(results.summary())
        factor_pvals = zip(results.pvalues.index, results.pvalues.values)
        for factor_pval in factor_pvals:
            factor, pvalue = factor_pval
            if pvalue < SIGNIF_LEVEL:
                if (portfolios.get(str(factor))) is None:
                    portfolios[str(factor)] = set(stock)
                else:
                    portfolios[str(factor)].add(stock)

    return portfolios


def debug_shape(dfs: list[pd.DataFrame]):
    for df in dfs:
        print(df.shape)


def add_factors_from_csv(directory) -> pd.DataFrame:
    """Takes a factor name and a CSV file, calculates the returns and returns them as a dataframe

    The first two elements should be a date column heading and a "Close" (case sensitive) column heading. The data should be two columns corresponding to dates and closing prices. 
    This method is untested and may need to be modified.
    """
    factorlist: pd.DataFrame = []
    for file in os.listdir(directory):
        filepath = directory + file
        factor_close = pd.read_csv(filepath, index_col=0, parse_dates=['Date'])
        # factor_close = pd.read_csv(filepath)
        factorlist.append(factor_close)

    combined_factors: pd.DataFrame = factorlist[0]
    # combined_factors['Date'] = pd.to_datetime(combined_factors['Date'])
    for i in range(1, len(factorlist)):
        df = factorlist[i]
        # df['Date'] = pd.to_datetime(df['Date'])
        combined_factors = combined_factors.join(df, on='Date', how='left', lsuffix='_left', rsuffix='_right')
        # combined_factors = combined_factors.set_index('Date').join(df.set_index('Date'), on='Date')

    return combined_factors.apply(lambda factor: get_returns(factor))


def normalizeFactorDates(dates: pd.DataFrame, stock_factors: pd.DataFrame) -> pd.DataFrame:
    return dates.join(stock_factors, on='Date', how='left', lsuffix='_left', rsuffix='_right')


kwargs = {'interval': '1mo'}
# factors = add_factors_from_csv('../factorDirectory/')

ticker_list = ['NFLX'] # still need to add support for multiple tickers
tickers = set()
for ticker in ticker_list:
    tickers.add(ticker)

stocks = add_stocks_from_tickers(tickers)
# dates = stocks.index.to_frame().reset_index(drop=True)
# print(dates)
factors = add_factors_from_csv('factorDirectory/')

# factors.to_csv('someshit.csv')
normalizedFactors = normalizeFactorDates(stocks, factors)
for i in range(0, len(normalizedFactors.columns) - 1):
    if len(normalizedFactors.columns[i]) == 2:
        normalizedFactors = normalizedFactors.drop(normalizedFactors.columns[i], axis=1)
    #     can't drop more because the stupid changes indexs when we drop stuff - works for 1 ticker
    # normalizedFactors.drop(columns=[str(column)])

# normalizedFactors.drop(columns=['(\'Close\', \'NFLX\')'])
# print(normalizedFactors.columns)
# print('------------------------------')
# print(stocks.columns)
# normalizedFactors.drop(columns=stocks.columns)
normalizedFactors.to_csv('bitching.csv')

# stocks.to_csv('../StockDf.csv')
# sml = add_factors_from_csv('./SML.csv')

# debug_shape([stock_returns, factors])

# portfolios = regress_factors(stock_returns, factors)
# print(portfolios)
