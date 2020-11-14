# Some of the implementation inspired by:
# REF: https://github.com/fchen365/epca

import numpy as np
from factor_analyzer import Rotator

from graspologic.embed import selectSVD
from ..utils import soft_threshold

def _varimax(X):
    return Rotator(normalize=False).fit_transform(X)


def _polar(X):
    # REF: https://en.wikipedia.org/wiki/Polar_decomposition#Relation_to_the_SVD
    U, D, Vt = selectSVD(X, n_components=X.shape[1], algorithm="full")
    return U @ Vt


def _polar_rotate_shrink(X, gamma=0.1):
    U, D, Vt = selectSVD(X, n_components=X.shape[1], algorithm="full")
    U_rot = _varimax(U)
    U_thresh = soft_threshold(U_rot, gamma)
    return U_thresh


def _reorder_components(X, Z_hat, Y_hat):
    score_norms = np.linalg.norm(X @ Y_hat, axis=0)
    sort_inds = np.argsort(-score_norms)
    return Z_hat[:, sort_inds], Y_hat[:, sort_inds]


def sparse_component_analysis(
    X, n_components=2, gamma=None, max_iter=10, reorder_components=True
):
    X = X.copy()
    U, D, Vt = selectSVD(X, n_components=n_components)
    if gamma is None:
        gamma = np.sqrt(U.shape[1] * X.shape[1])
    Z_hat = U
    Y_hat = Vt.T
    i = 0
    while i < max_iter:
        Y_hat = _polar_rotate_shrink(X.T @ Z_hat, gamma=gamma)
        Z_hat = _polar(X @ Y_hat)
        i += 1
    if reorder_components:
        Z_hat, Y_hat = _reorder_components(X, Z_hat, Y_hat)
    return Z_hat, Y_hat
\
