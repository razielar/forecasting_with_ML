# Forecasting with Machine Learning

Repo from the course is placed [here](https://github.com/trainindata/forecasting-with-machine-learning).     
Additionally, Feature Engineering for Time Series Forecasting (course 1) is placed [here](https://github.com/razielar/feature_engineering_ts_forecasting?tab=readme-ov-file).

1. [Time Series as regression](#one)
2. [Multistep forecasting](#two)
3. [Multiseries forecasting](#three)
4. [Backtesting](#four)
5. [Error metrics](#five)

## 1) <a id='one'></a> Time Series as regression

### Time series types

Regular types are easier to forecast them.

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/time_series_types.png" alt="logo"></img>
</div>

### Time series components

* **Trend**: long term change in the mean of the time series.
* **Seasonality**: regular, repetitive fluctuations.
* **Cyclicity**: irregular fluctuations over longer time periods. For example, recessions or wars.
* **Residuals**: error term.

### Forecasting models

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/forecasting_models.png" alt="logo"></img>
</div>

Some advantages of using ML models: 
* Interpretability
* Model non-linear or complex relationships
* Model multiple time series simultaneously
* Combine features from: the time series and exogenous data sources

### Benchmark models

| Baseline model | Description | 
|----------|----------|
| **Naive forecast**          | Uses the most recent observation as forecast.                           | 
| **Moving average**          | Uses the mean of the X most recent observations as forecast.            | 
| **Seasonal naive forecast** | Uses the most recent observation within the same season as forecast.    | 
| **Historial average**       | Uses the average of historical data as forecast.                        | 

## 2) <a id='two'></a> Multistep forecasting

### Recursive forecast

* Train a single model to predict one-step ahead.
* Use the model recursively, to predict each step of the horizon.
* Oldest, most intuitive and most popular strategy.
* It is incorporated in statatistical forecasting models such as: ARIMA and ETS.

| Pros | Cons |
|----------|----------|
| One ML model   | Bias and variance of the error in the first step of forecast accumulate as we move further in the horizon. |

### Direct forecast

## 3) <a id='three'></a> Multiseries forecasting

## 4) <a id='four'></a> Backtesting

## 5) <a id='five'></a> Error metrics
