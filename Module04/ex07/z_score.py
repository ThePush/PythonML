import numpy as np


def zscore_(x, mean, std):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return
    return (x - mean) / std


def zscore(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return
    return (x - x.mean(axis=0)) / x.std(axis=0)
