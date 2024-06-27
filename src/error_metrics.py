# Error metrics to measure:
    # 1) Bias
    # 2) Error

import numpy as np

def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Forecast bias: is a measure of forecast bias (i.e. over or under forecasting)
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Forecast predictions
    Returns:
        float: forecast bias
    """
    sum_errors = np.sum(y_true - y_pred)
    sum_y = np.sum(y_true)
    return sum_errors / sum_y

