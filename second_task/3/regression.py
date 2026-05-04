from typing import Optional, Self
import numpy as np


def TSS(y):
    return np.sum((y - y.mean()) ** 2)


def RSS(y, y_pred):
    return np.sum((y - y_pred) ** 2)


def R_square(y, y_pred):
    tss_y = TSS(y)
    rss_y = RSS(y, y_pred)
    return (tss_y - rss_y) / tss_y


class LinearRegression:
    # regression coefficients
    _beta: Optional[np.ndarray]
    _bias: Optional[np.float64]

    # Fisher matrix
    _F_inv: np.ndarray

    # is bias neccesary
    _fit_bias: bool

    def __init__(
        self,
        fit_bias: bool = True,
    ) -> None:
        self._beta = self._bias = None
        self._fit_bias = fit_bias

    @staticmethod
    def _is_positive_defined(A, tol=1e-8):
        E = np.linalg.eigvalsh(A)
        return np.all(E > -tol)

    def _check_F(
        self,
        F: np.ndarray
    ) -> None:
        if not self._is_positive_defined(F):
            raise RuntimeError("F must be positive defined!")

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> Self:
        if self._fit_bias:
            psi = np.concat((X, np.ones(X.shape[0]).reshape(-1, 1)), axis=1)
        else:
            psi = X

        F = psi.T @ psi
        self._check_F(F)
        self._F_inv = np.linalg.inv(F)

        self._beta = (self._F_inv @ psi.T @ Y).flatten()

        if self._fit_bias:
            beta_indexes = np.ones(X.shape[1] + 1, dtype=np.int_)
            beta_indexes[-1] = 0

            self._bias = self._beta[-1]
            self._beta = self._beta[:-1]

        else:
            self._bias = np.float64(0)

        return self

    def predict(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        if self._beta is None or self._bias is None:
            raise RuntimeError("Model is unfitted!")
        return x @ self._beta + self._bias

    @property
    def Fisher_matrix(
        self
    ) -> np.ndarray:
        return self._F_inv

    @property
    def coeff(
        self
    ) -> np.ndarray:
        if self._beta is None or self._bias is None:
            raise RuntimeError("Model is unfitted!")
        return (
            np.concat((self._beta, [self._bias])) if self._fit_bias else
            self._beta
        )
