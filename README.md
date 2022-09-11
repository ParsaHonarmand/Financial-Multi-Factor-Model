# Financial Analysis in Python

## Instructions
### To run linear regression:
  
  `cd src`

  `python3 linear_regression.py`

Results will be output to terminal


### To run machine learning model: 

  `cd src`

  `python3 gru.py`

Results will be in the neuralNetResults directory

## Overview
As stock markets and economies have evolved, economists and investors have continuously attempted to improve their stock return prediction models in search of sustainable investment profits. We wanted to investigate the factor investing approach to financial modelling. This approach involves targeting quantifiable firm characteristics or “factors” that help predict stock returns. This is considered a sophisticated investment method used by institutional investors such as pensions and hedge funds. We sampled a large number of stocks in the S&P 500 (S&P), a portfolio of the biggest public companies in the U.S that acts as a proxy for the whole market.

## Linear Regression Model
We chose a set of 11 factors that we believed were most appropriate to explain general stock returns, ranging from volatility level to inverse correlation with interest rates, and we iteratively ran numerous multiple linear regressions with the different factors being the set of independent variables and the stock returns being the dependent variable. We checked for factors that had a regression p-value of over 0.05 (signalling non-statistical significance), removed them from the regression, and re-ran the regression without said factors. The result of our model is every combination of the 11 factors and the stocks whose price they can predict to a statistically significant degree. 

## Optimizations
To address perfomance bottlenecks, we partitioned the data so that we could run each linear regression in a separate Apache Spark resilient distributed dataset. With 8 partitions, we were able to reduce the time to run our linear regression from 351 seconds down to 45 seconds, a 7.8x improvement. 

## Gated Recurrent Network Model
Linear regression is a naive statistical approach for predicting stock prices. We decided to build a Gated Recurrent Network neural network to contrast the naive linear regression approach with a more sophisticated neural network. The neural network used only historical stock prices as the inputs. From financial theory, historical stock prices should not have an impact on future performance. Thus, it was surprising to observe that the neural network prediction results were comparable to those of the linear regression model. The mean squared error for the neural network was ~1%, which is on par with that of the multiple linear regression.

### Full Report
More details about the project can be found in the Multifactor Financial Model Report PDF in this repository. 
