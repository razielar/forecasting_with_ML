# Forecasting with Machine Learning

Repo from the course is placed [here](https://github.com/trainindata/forecasting-with-machine-learning).     
Additionally, Feature Engineering for Time Series Forecasting (course 1) is placed [here](https://github.com/razielar/feature_engineering_ts_forecasting?tab=readme-ov-file).

1. [Time Series as regression](#one)
2. [Multistep forecasting](#two)
3. [Multiseries forecasting](#three)
4. [Backtesting](#four)
5. [Error metrics](#five)
6. [Bibliography](#six)

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

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/recursive_forecast_mod.png" alt="logo"></img>
</div>

* Train a single model to predict one-step ahead.
* Use the model recursively, to predict each step of the horizon.
* Oldest, most intuitive and most popular strategy.
* It is incorporated in statatistical forecasting models such as: ARIMA and ETS.

| Pros | Cons |
|----------|----------|
| One ML model   | Bias and variance of the error in the first step of forecast accumulate as we move further in the horizon. |

### Direct forecast

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/direct_forecast.png" alt="logo"></img>
</div>

* Train one model per step in the horizon.
* Each step in the horizon is forecast independently.

| Pros | Cons |
|----------|----------|
| Does not propagate estimation errors. | Forecast are independent. |
| Less estimation variance              | Computational cost. |

## 3) <a id='three'></a> Multiseries forecasting

### Multiple time series

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/multiple_ts.png" alt="logo"></img>
</div>

Examples of **multiple independent time series**:
* Energy demand by household.
* Sale products in different stores.
* Air pollution in different cities.
* Picked SKUs (stock keeping units) by warehouse.

Examples of **multiple dependent time series** or **multivariate time series**:
* The simultaneous measurement of: blood preassure, temperature, and heart rate of the same patient.
* Measurement of multiple sensors such as: temperature, preassure, and flow.

### Global and Local forecasting

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/global_forecasting.png" alt="logo"></img>
</div>

| Num  | Global forecasting Pros | Global forecasting Cons |
|------|----------|----------|
| 1    | Easier to matain a single model.                                | Without a shared global pattern, the model may lear a pattern that isn't representative. |
| 2    | Learns common patterns across the whole dateset.                | The above depends on: model complexity, features and heterogeneity of the time series.   |
| 3    | Increases the sample size when we pass multiple ts to the model | Training and backtesting larger models can take more time.                               |
| 4    | Better at forecasting a short time series.                      |    |


<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/local_forecasting.png" alt="logo"></img>
</div>

| Num  | Local forecasting Pros | Local forecasting Cons |
|------|----------|----------|
| 1    | Each model can adjust its parameters to fit each time series individually. | Worse approach on short time series, can't learn much from small sample size. |
| 2    |    | Computational and maitenance cost of using one model per series.     |
| 3    |    | Does not use all the learnable information available in the dateset. |

### Global vs. Local forecasting

| Num | Global models | Local models |
|-----|----------|----------|
| 1   | A large number of related time series.                                   | A small number of time series.                       |
| 2   | The time series share some global patterns.                              | Each time series has very different characterictics. |
| 3   | We have exogenous features that impact many of the time series.          | |
| 4   | We want to forecast shorter time series after learning from multiple ts. | |
| 5   | Use all relevant features and complex models to capture the variety of patterns that exist in the dataset.                             | |
| 6   | Can try training multiple global models on subgroups of the data if there is a lot of different characteristics in the ts by subgroup. | |

## 4) <a id='four'></a> Backtesting

## 5) <a id='five'></a> Error metrics

**Error metrics**: summarize forecast errors into a single number to measure the accuracy of a forecast.    
We need to transform forecasting errors to make them positive, to ensure negative and positive errors don't cancel each other. In general, we can convert forecasting error by:
* **Absolute** transformation: *e.g.* Mean absolute error which optimeze for the median thus better for ts with outliers.
* **Square** transformation: *e.g.* Root mean square error which optimeze for the mean thus better for intermittent ts.

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right| $$

$$ \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 } $$

### Error metrics classification

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/error_metrics_table.png" alt="logo"></img>
</div>

As described above, error metrics can be classified by:
* Aggregation: mean or median (more robust to outliers).
* Type of forecast error transformation: absolute or square transformation.
* Other. 

### Measuring forecasting errors

Factors that impact which error metric to use:

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/factor_error_metrics.png" alt="logo"></img>
</div>

### Multiple time series




### Measuring forecasting bias

`Bias`: is where we on average, over or under forecast.

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/forecast_bias.png" alt="logo"></img>
</div>

The model 2 (red bins): `under forecast`.

**Metrics to measure bias**:

| Num | Name | Desc |
|----------|----------|----------|
| 1   | Mean error (ME)                 | scale dependent     |
| 2   | Cumulative forecast error (CFE) | scale dependent     |
| 3   | Forecast bias  (FB)             | scale independent   |
| 4   | Tracking signal ($TS_w$)        | scale independent, w: past rolling window. Threshold (+- 3.75) |

## 6) <a id='six'></a> Bibliography

* **Error metrics**: [Forecast evaluation for data scientists: common pitfalls and best practices](https://link.springer.com/article/10.1007/s10618-022-00894-5)
