# Some of the implementation inspired by:
# REF: https://github.com/fchen365/epca

import time
from abc import abstractmethod

import numpy as np
from factor_analyzer import Rotator
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from graspologic.embed import selectSVD

from ..utils import calculate_explained_variance_ratio, soft_threshold

from scipy.linalg import orthogonal_procrustes


def _varimax(X):
    return Rotator(normalize=False).fit_transform(X)


def _polar(X):
    # REF: https://en.wikipedia.org/wiki/Polar_decomposition#Relation_to_the_SVD
    U, D, Vt = selectSVD(X, n_components=X.shape[1], algorithm="full")
    return U @ Vt


def _polar_rotate_shrink(X, gamma=0.1):
    # Algorithm 1 from the paper
    U, _, _ = selectSVD(X, n_components=X.shape[1], algorithm="full")
    # U = _polar(X)
    # R, _ = orthogonal_procrustes(U_old, U)
    # print(np.linalg.norm(U_old @ R - U))
    U_rot = _varimax(U)
    U_thresh = soft_threshold(U_rot, gamma)
    return U_thresh


def _reorder_components(X, Z_hat, Y_hat):
    score_norms = np.linalg.norm(X @ Y_hat, axis=0)
    sort_inds = np.argsort(-score_norms)
    return Z_hat[:, sort_inds], Y_hat[:, sort_inds]


# import abc


# class SuperclassMeta(type):
#     def __new__(mcls, classname, bases, cls_dict):
#         cls = super().__new__(mcls, classname, bases, cls_dict)
#         for name, member in cls_dict.items():
#             if not getattr(member, "__doc__"):
#                 member.__doc__ = getattr(bases[-1], name).__doc__
#         return cls


class BaseSparseDecomposition(BaseEstimator):
    def __init__(
        self,
        n_components=2,
        gamma=None,
        max_iter=10,
        scale=False,
        center=False,
        tol=1e-4,
        verbose=0,
    ):
        """Sparse matrix decomposition model.

        Parameters
        ----------
        n_components : int, optional (default=2)
            Number of components or embedding dimensions.
        gamma : float, int or None, optional (default=None)
            Sparsity parameter, must be nonnegative. Lower values lead to more sparsity
            in the estimated components. If ``None``, will be set to 
            ``sqrt(n_components * X.shape[1])`` where ``X`` is the matrix passed to 
            ``fit``.
        max_iter : int, optional (default=10)
            Maximum number of iterations allowed, must be nonnegative.
        scale : bool, optional
            [description], by default False
        center : bool, optional
            [description], by default False
        tol : float or int, optional (default=1e-4)
            Tolerance for stopping iterative optimization. If the relative difference in
            score is less than this amount the algorithm will terminate.
        verbose : int, optional (default=0)
            Verbosity level. Higher values will result in more messages. 
        """
        self.n_components = n_components
        self.gamma = gamma
        self.max_iter = max_iter
        self.scale = scale
        self.center = center
        self.tol = tol
        self.verbose = verbose
        # TODO add random state

    def _initialize(self, X):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        U, D, Vt = selectSVD(X, n_components=self.n_components)
        score = np.linalg.norm(D)
        return U, Vt.T, score

    def _validate_parameters(self, X):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        """
        if not self.gamma:
            gamma = np.sqrt(self.n_components * X.shape[1])
        else:
            gamma = self.gamma
        self.gamma_ = gamma

    def _preprocess_data(self, X):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        if self.scale or self.center:
            X = StandardScaler(
                with_mean=self.center, with_std=self.scale
            ).fit_transform(X)
        return X

    # def _compute_matrix_difference(X, metric='max'):
    # TODO better convergence criteria

    def fit_transform(self, X, y=None):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        y : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        self._validate_parameters(X)

        self._validate_data(X, copy=True, ensure_2d=True)  # from sklearn BaseEstimator

        Z_hat, Y_hat, score = self._initialize(X)

        if self.gamma == np.inf:
            max_iter = 0
        else:
            max_iter = self.max_iter

        # for keeping track of progress over iteration
        Z_diff = np.inf
        Y_diff = np.inf
        norm_score_diff = np.inf
        last_score = 0

        # main loop
        i = 0
        while (i < max_iter) and (norm_score_diff > self.tol):
            if self.verbose > 0:
                print(f"Iteration: {i}")

            iter_time = time.time()

            Z_hat_new, Y_hat_new = self._update_estimates(X, Z_hat, Y_hat)

            # Z_hat_new, Y_hat_new = _reorder_components(X, Z_hat_new, Y_hat_new)
            Z_diff = np.linalg.norm(Z_hat_new - Z_hat)
            Y_diff = np.linalg.norm(Y_hat_new - Y_hat)
            norm_Z_diff = Z_diff / np.linalg.norm(Z_hat_new)
            norm_Y_diff = Y_diff / np.linalg.norm(Y_hat_new)

            Z_hat = Z_hat_new
            Y_hat = Y_hat_new

            B_hat = Z_hat.T @ X @ Y_hat
            score = np.linalg.norm(B_hat)
            norm_score_diff = np.abs(score - last_score) / score
            last_score = score

            if self.verbose > 1:
                print(f"{time.time() - iter_time:.3f} seconds elapsed for iteration.")

            if self.verbose > 0:
                print(f"Difference in Z_hat: {Z_diff}")
                print(f"Difference in Y_hat: {Z_diff}")
                print(f"Normalized difference in Z_hat: {norm_Z_diff}")
                print(f"Normalized difference in Y_hat: {norm_Y_diff}")
                print(f"Total score: {score}")
                print(f"Normalized difference in score: {norm_score_diff}")
                print()

            i += 1

        Z_hat, Y_hat = _reorder_components(X, Z_hat, Y_hat)

        # save attributes
        self.n_iter_ = i
        self.components_ = Y_hat.T
        # TODO this should not be cumulative by the sklearn definition
        self.explained_variance_ratio_ = calculate_explained_variance_ratio(X, Y_hat)
        self.score_ = score

        return Z_hat

    def fit(self, X):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        self.fit_transform(X)
        return self

    def transform(self, X):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        # TODO input checking
        return X @ self.components_.T

    @abstractmethod
    def _update_estimates(self, X, Z_hat, Y_hat):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        Z_hat : [type]
            [description]
        Y_hat : [type]
            [description]
        """
        pass


class SparseComponentAnalysis(BaseSparseDecomposition):
    def _update_estimates(self, X, Z_hat, Y_hat):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        Z_hat : [type]
            [description]
        Y_hat : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        Y_hat = _polar_rotate_shrink(X.T @ Z_hat, gamma=self.gamma)
        Z_hat = _polar(X @ Y_hat)
        return Z_hat, Y_hat

    def _save_attributes(self, X, Z_hat, Y_hat):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        Z_hat : [type]
            [description]
        Y_hat : [type]
            [description]
        """
        pass


class SparseMatrixApproximation(BaseSparseDecomposition):
    def _update_estimates(self, X, Z_hat, Y_hat):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        Z_hat : [type]
            [description]
        Y_hat : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        Z_hat = _polar_rotate_shrink(X @ Y_hat)
        Y_hat = _polar_rotate_shrink(X.T @ Z_hat)
        return Z_hat, Y_hat

    def _save_attributes(self, X, Z_hat, Y_hat):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        Z_hat : [type]
            [description]
        Y_hat : [type]
            [description]
        """
        B = Z_hat.T @ X @ Y_hat
        self.score_ = B
        self.right_latent_ = Y_hat
        self.left_latent_ = Z_hat
