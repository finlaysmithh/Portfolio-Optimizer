from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    signs = np.sign(y_true) == np.sign(y_pred)
    return float(np.nanmean(signs.astype(float)))


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    # Approximate CRPS for Gaussian forecasts (closed-form)
    from scipy.stats import norm

    z = (y_true - mu) / (sigma + 1e-8)
    return float(np.nanmean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))

