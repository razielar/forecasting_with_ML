# Functions for 2) multistep forecasting
import numpy as np

# To add window features, we need to create a function that creates those features
# For the moment, we cannot use: LagFeatures from Feature Engine. Because,
#   skforecast treats the time-series as a numpy vector instead of pandas (Feature engine)
def create_window_features(y: np.array) -> np.array:
    """In this function, we create:
        Lag: we create 10 lags, from lag1 to lag10
        Window features: mean with window size of 3 and 24
    Args:
        y (_type_): _description_
    Returns:
        _type_: _description_
    """
    # lag features 
    lags = y[-1:-11:-1] # Here, we create 10 lags, from lag1 to lag10
    # lag_5 = y[-5:-4] # Example of creating a specific lag
    
    # window features
    mean_3 = np.mean(y[-3:])   # window_size = 3
    mean_24 = np.mean(y[-24:]) # window_size = 24
    # put all together
    predictors = np.hstack([lags, mean_3, mean_24])
    return predictors