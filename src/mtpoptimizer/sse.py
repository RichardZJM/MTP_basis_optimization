"""
Module for computing the accuracy, the Sum of Squared Errors (SSE).

This module provides functionality to efficiently compute SSE values for pruned MTP feature sets. It uses pre-computed matrices to avoid redundant calculations and supports Tikhonov regularization for numerical stability.
"""

from typing import Tuple, Union
import numpy as np
import warnings


class SSECalculator:
    """
    Calculator for Sum of Squared Errors.

    This class provides efficient computation of SSE values using pre-computed matrices and optional regularization.  It warns about ill-conditioned systems.
    """

    def __init__(
        self,
        xtwx: np.ndarray,
        xtwy: np.ndarray,
        ytwy: float,
        regularization: float = 0.0,
        rank: int = 0,
    ):
        """
        Initialize SSE calculator with pre-computed matrices.

        Parameters
        ----------
        xtwx : np.ndarray
            Pre-computed XᵀWX matrix, shape (n_features, n_features)
        xtwy : np.ndarray
            Pre-computed XᵀWy vector, shape (n_features,)
        ytwy : float
            Pre-computed yᵀWy scalar
        regularization : float, optional
            Tikhonov regularization parameter (λ), by default 0.0.
            Adds λI to XᵀWX for numerical stability
        rank : int, optional
            MPI rank of the process (for controlling output), by default 0
        """

        self.xtwx = np.asarray(xtwx)
        self.xtwy = np.asarray(xtwy)
        self.ytwy = np.asarray(ytwy)

        # Apply Tikihonov regularization
        self.xtwx = self.xtwx + regularization * np.eye(len(self.xtwy))

        self.base_sse = 1
        self.base_sse = self.calculate(np.ones_like(self.xtwy).astype(bool))

        if rank == 0:
            print(f"The base SSE is {self.base_sse:.2f}.")

        cond = np.linalg.cond(self.xtwx)

        if cond > 1e10 and rank == 0:
            warnings.warn(
                f"Matrix is ill-conditioned: {cond:.2e} > 1e10 . Please consider more regularization."
            )

    def calculate(
        self, mask: np.ndarray, get_theta: bool = False
    ) -> Union[float, Tuple[np.ndarray, float]]:
        """
        Calculates the normalized cost heuristic for a pruned feature set.

        Computes the Sum of Squared Errors for a subset of features defined by the mask including species coefficients.  Optionally returns the fitted coefficients along with the SSE value.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask indicating which basis functions to include. Must include species coefficients.
        get_theta : bool, optional
            Whether to return the fitted coefficients, by default False

        Returns
        -------
        Union[float, Tuple[np.ndarray, float]]
            If get_theta=False:
                Normalized SSE value
            If get_theta=True:
                Tuple of (fitted coefficients, normalized SSE)
            Returns inf if the system is singular
        """
        # Apply the mask to the pre-calculated matrices
        xtwxm = self.xtwx[mask][:, mask]
        xtwym = self.xtwy[mask]

        try:
            theta = np.linalg.solve(xtwxm, xtwym)
            # theta = sp.linalg.solve(xtwxm, xtwym, assume_a="pos")
            # L = np.linalg.cholesky(xtwxm)

            # y = np.linalg.solve(L, xtwym)
            # theta = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            # If the matrix is singular, the system can't be solved.
            return float("inf")

        sse = (self.ytwy - theta @ xtwym) / self.base_sse

        # We must use item since it may be a single element numpy array
        if get_theta:
            return (theta, sse.item())
        return sse.item()
