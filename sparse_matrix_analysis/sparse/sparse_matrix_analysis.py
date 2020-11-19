# Some of the implementation inspired by:
# REF: https://github.com/fchen365/epca

from abc import abstractmethod

import numpy as np
from factor_analyzer import Rotator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

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
    X,
    n_components=2,
    gamma=None,
    max_iter=10,
    reorder_components=True,
    scale=False,
    center=False,
):
    # TODO standard scaler
    # TODO center
    X = X.copy()
    if scale or center:
        X = StandardScaler(with_mean=center, with_std=scale).fit_transform(X)
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


# def sparse_matrix_approximation(
#     X,
#     n_components=2,
#     gamma=None,
#     max_iter=10,
#     reorder_components=True,
#     scale=False,
#     center=False,
# ):


class BaseSparseDecomposition(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=2,
        gamma=None,
        max_iter=10,
        reorder_components=True,
        scale=False,
        center=False,
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.max_iter = max_iter
        self.reorder_components = reorder_components
        self.scale = scale
        self.center = center

    def _initialize(self, X):
        U, D, Vt = selectSVD(X, n_components=self.n_components)
        return U, Vt.T

    def _validate_parameters(self, X):
        if not self.gamma:
            # TODO not sure if this should be shape[1] or shape[0]
            gamma = np.sqrt(self.n_components * X.shape[1])
        else:
            gamma = self.gamma
        self.gamma_ = gamma

    def _preprocess_data(self, X):
        if self.scale or self.center:
            X = StandardScaler(
                with_mean=self.center, with_std=self.scale
            ).fit_transform(X)
        return X

    def fit_transform(self, X, y=None):
        self._validate_parameters(X)

        self._validate_data(X, copy=True, ensure_2d=True)

        Z_hat, Y_hat = self._initialize(X)

        # main loop
        i = 0
        while i < self.max_iter:  # TODO other stopping criteria
            Z_hat, Y_hat = self._update_estimates(X, Z_hat, Y_hat)
            i += 1

        if self.reorder_components:
            Z_hat, Y_hat = _reorder_components(X, Z_hat, Y_hat)

        self._save_attributes(X, Z_hat, Y_hat)

        return Z_hat

    def _save_attributes(self, X, Z_hat, Y_hat):
        self.components_ = Y_hat.T
        # TODO compute PVE

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X):
        # TODO input checking
        return X @ self.components_.T

    @abstractmethod
    def _update_estimates(self, X, Z_hat, Y_hat):
        pass


class SparseComponentAnalysis(BaseSparseDecomposition):
    def _update_estimates(self, X, Z_hat, Y_hat):
        Y_hat = _polar_rotate_shrink(X.T @ Z_hat, gamma=self.gamma)
        Z_hat = _polar(X @ Y_hat)
        return Z_hat, Y_hat


class SparseMatrixApproximation(BaseSparseDecomposition):
    def _update_estimates(self, X, Z_hat, Y_hat):
        Z_hat = _polar_rotate_shrink(X @ Y_hat)
        Y_hat = _polar_rotate_shrink(X.T @ Z_hat)
        return Z_hat, Y_hat

    def _save_attributes(self, X, Z_hat, Y_hat):
        B = Z_hat.T @ X @ Y_hat
        self.scores_ = B
        self.right_latent_ = Y_hat
        self.left_latent_ = Z_hat

