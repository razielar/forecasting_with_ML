# Functions for 3) multiseries forecasting
import numpy as np

def create_predictors(y: np.array) -> np.array:
    """
    Create first 7 lags of a time series.
    Calculate rolling mean and standard deviation with window 7 and 30.
    """
    lags = y[-1:-8:-1]  # window_size = 8 => 7 lags
    mean_7 = np.mean(y[-7:])  # window_size = 7
    std_7 = np.std(y[-7:])
    mean_30 = np.mean(y[-30:])  # window_size = 30
    std_30 = np.std(y[-30:])
    predictors = np.hstack([lags, mean_7, std_7, mean_30, std_30])
    return predictors
