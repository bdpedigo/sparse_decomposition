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


def proportion_variance_explained(X, Y, center=False):
    if center:
        X = X.copy()
        X -= X.mean(axis=0)[None, :]
    return (np.linalg.norm(X @ Y @ np.linalg.inv(Y.T @ Y) @ Y.T, ord="fro") ** 2) / (
        np.linalg.norm(X, ord="fro") ** 2
    )


def calculate_explained_variance_ratio(X, Y, center=False):
    n_components = Y.shape[1]
    explained_variance_ratio = []
    for i in range(1, n_components + 1):
        pve = proportion_variance_explained(X, Y[:, :i])
        explained_variance_ratio.append(pve)
    return np.array(explained_variance_ratio)
