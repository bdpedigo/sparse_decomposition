import numpy as np


def threshold_array(X, threshold):
    """Soft thresholding or shrinkage operator"""
    X_thresh = np.sign(X) * np.maximum(np.abs(X) - threshold, 0)
    return X_thresh


def l1_norm(X):
    return np.abs(X).sum()


def soft_threshold(X, gamma=None, eps=1e-11):
    if not gamma:
        gamma = np.sqrt(X.shape[0] * X.shape[1])

    if l1_norm(X) < gamma:
        return X

    lower = 0
    upper = np.max(np.abs(X))
    while (upper - lower) > eps:
        mid = (lower + upper) / 2
        mid_norm = l1_norm(threshold_array(X, mid))
        if mid_norm > gamma:
            lower = mid
        else:
            upper = mid
    X_thresh = threshold_array(X, mid)
    return X_thresh
