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

| Baseline model | Description | Comments |
|----------|----------|----------|
| Naive forecast          | Uses the most recent observation as forecast.                           | Row 1    |
| Moving average          | Uses the mean of the X most recent observations as forecast.            | Row 2    |
| Seasonal naive forecast | Uses the most recent observation within the same season as forecast.    | Row 3    |
| Historial average       | Uses the average of historical data as forecast.                        | Row 4    |

## 2) <a id='two'></a> Multistep forecasting

## 3) <a id='three'></a> Multiseries forecasting

## 4) <a id='four'></a> Backtesting

## 5) <a id='five'></a> Error metrics
