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

Backtesting is where we measure the performance of a forecasting model on historic data. Backtesting is like: **cross validation** but for time series. Often used for: **a)** Model selection, **b)** Feature selection, **c)** Hyper-paramer tuning.

Before, starting we need to define the following variables: Dataset, Forecasting horizon, Error metrics, and Models. 
Additionally, we need to consider the following:
* Training window: initial traing size and expanding or rolling.
* Model refitting: once, multiple times or intermittent.
* Number of steps the forecasting origin moves.
* Gap: gap or not gap.

**Ranking of backtesting strategies** (from more generalized to less generalizable):

* **1)** Backtesting with refit and fixed training size.
* **2)** Backtesting with intermittent refitting.
* **3)** Backtesting with refit (`Nixtla` strategy).
* **4)** Backtesting without refit.

### Backtesting with refit and fixed training size

Also, named: Backtesting with refit and rolling training window.

```python
backtesting_forecaster(
    fixed_train_size=True,
    refit=True,
)
```

![image](https://github.com/razielar/forecasting_with_ML/blob/main/img/backtesting/backtesting_refit_fixed_train_size.gif)

### Backtesting with refit

```python
backtesting_forecaster(
    fixed_train_size=False,
    refit=True,
)
```

![image](https://github.com/razielar/forecasting_with_ML/blob/main/img/backtesting/backtesting_refit.gif)

### Backtesting without refit

Genrally avoid use it.

![image](https://github.com/razielar/forecasting_with_ML/blob/main/img/backtesting/backtesting_no_refit.gif)


## 5) <a id='five'></a> Error metrics

**Error metrics**: summarize forecast errors into a single number to measure the accuracy of a forecast.    
We need to transform forecasting errors to make them positive, to ensure negative and positive errors don't cancel each other. In general, we can convert forecasting error by:
* **Absolute** transformation: *e.g.* Mean absolute error which optimeze for the median thus better for ts with outliers.
* **Square** transformation: *e.g.* Root mean square error which optimeze for the mean thus better for intermittent ts.

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right| $$

$$ \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 } $$

To see a complitec guideline for when to use each error metric, see: [error metric guidelines](https://github.com/razielar/forecasting_with_ML/blob/main/05_error_metrics/Guidelines.md).

### Error metrics classification

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/error_metrics_table.png" alt="logo"></img>
</div>

As described above, error metrics can be classified by:
* Aggregation: mean or median (more robust to outliers).
* Type of forecast error transformation: absolute or square transformation.
* Other. 

Factors that impact which error metric to use:

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/factor_error_metrics.png" alt="logo"></img>
</div>

* **1)** **Outliers**: if present use absolute errors (MAE) or median as aggregation method.
* **2)** **Multiple time series**: decide if use scale-dependent (we care more for ts with higher sales) or scale-independent (we care equally to all time series) metric.
* **3)** **Intermittent time series**: squared errors (RMSE) tends to optimize on the mean, thus squared errors are better suited for intermittent ts.

### Scale dependent error metrics

MAE and MSE or RMSE are symmetric metrics *i.e.* symmetric to under and over forecasting, as depicted below:

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/symmetry_mae_mse.png" alt="logo"></img>
</div>


### Scale independent error metrics

**Percentage**

We rescale by the actual value, as described below:

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/percentage_scaling.png" alt="logo"></img>
</div>

$$ e_t = y_t - \hat{y}_t $$

Percentage error:

$$ p_t = \frac{100e_t}{y_t} $$

Symmetric percentage error:

$$ p^{*}_{t} = \frac{100e_t}{ \frac{1}{2} \left( |y_t| + |\hat{y_t}| \right) } $$

**NOTE**: sMAPE or symmetric MAPE its name is misleading is symmetric to: swapping real and predicted values. Nonetheless, not symmetric for: symmetric for over and under forecasting.

**Percentage error metrics**

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/percentage_error_metrics.png" alt="logo"></img>
</div>


Percentage errors behave badly both when $y = 0$ or $\hat{y} = 0$, as depicted in the plot for MAPE and sMAPE, respectively:

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/mape_smape_issue_plot.png" alt="logo"></img>
</div>

Further:
* MAPE: if we optime for MAPE, we `under forecast`.
* sMAPE: if we optime for sMAPE, we `under forecast`.

However, exist a modified percentage errors which help with zeros, see more [here](https://github.com/razielar/forecasting_with_ML/blob/main/05_error_metrics/Guidelines.md). Additionally,
`WAPE` (weighted mean absolute percentage error) is symmetric to over and under forecasting, can handle some zeros (unlike MAPE), but can't handle ts with all zeros and trend and level shift can be an issue for WAPE.

**Relative**

Using a baseline forecast as comparison, benchmark forecast: 
* Last value of the time series
* Historical average of the training set.

$$ e_t = y_t - \hat{y}_t $$

$$ e_t^b = y_t - \hat{y}_t^b $$

$$ r_t = \frac{e_t}{e_t^b}  $$

**Relative error metrics**

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/relative_error_metrics.png" alt="logo"></img>
</div>

**Scaled**

Introduced by Hyndman in 2005, 1-step or a seasonal naive forecast on the `training set`. Scaled errors are symmetric to over and under forecasting, are well behaved when actuals ($y$) and predictions ($\hat{y}$) are zero.   
However, errors in the `training set` may not be representative of the errors in the forecast horizon.

$$ e_t = y_t - \hat{y}_t $$

$$ q_t = \frac{e_t}{\text{MSE}_{\text{naive}}}  $$

$$ \text{MASE} =  \text{mean}(|q_t|) $$

$$ \text{RMSSE} =  \sqrt{ \text{mean}(q_t^2) } $$

Interpretation:
* When MASE or RMSSE > 1: performing worse than a naive forecast.
* When MASE or RMSSE < 1: performing better than a naive forecast.


### Multiple time series

<div align="center">
<img src="https://github.com/razielar/forecasting_with_ML/blob/main/img/multiple_ts_errors.png" alt="logo"></img>
</div>


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
