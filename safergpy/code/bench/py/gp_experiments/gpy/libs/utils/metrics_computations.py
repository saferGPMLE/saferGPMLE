import scipy.stats
import numpy as np
import math


def get_gaussian_alpha_coverage(y, mu_pred, var_pred, alpha):
    assert mu_pred.shape == var_pred.shape, "Shape issue"
    assert isinstance(alpha, float), "alpha must be float"

    assert np.all(var_pred > 0)

    lower = scipy.stats.norm.ppf((1 - alpha) / 2, loc=mu_pred, scale=np.vectorize(math.sqrt)(var_pred))
    upper = scipy.stats.norm.ppf(1 - (1 - alpha) / 2, loc=mu_pred, scale=np.vectorize(math.sqrt)(var_pred))

    assert (y.shape == lower.shape) and (y.shape == upper.shape), \
        "Shape issue : {}, {} and {}".format(y.shape, lower.shape, upper.shape)

    is_alpha_credible = np.logical_and(lower <= y, y <= upper).astype(float)

    return is_alpha_credible.mean()


def get_residuals(mu_pred, y):
    assert isinstance(mu_pred, np.ndarray) and isinstance(y, np.ndarray), 'Type issue'
    assert mu_pred.shape == y.shape and mu_pred.ndim == 1, 'Shape issue'
    residuals = y - mu_pred
    return residuals


def get_mse(mu_pred, y):
    residuals = get_residuals(mu_pred, y)
    return (residuals**2).mean()


def get_vse(mu_pred, y):
    residuals = get_residuals(mu_pred, y)
    return (residuals**2).var()


def get_mae(mu_pred, y):
    residuals = get_residuals(mu_pred, y)
    return (abs(residuals)).mean()


def get_scaled_mse(mu_pred, var_pred, y):
    residuals = get_residuals(mu_pred, y)

    assert isinstance(residuals, np.ndarray) and isinstance(var_pred, np.ndarray), 'Type issue'
    assert residuals.shape == var_pred.shape and var_pred.ndim == 1, 'Shape issue'

    assert np.all(var_pred > 0)

    return ((residuals**2) / var_pred).mean()
