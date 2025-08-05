import numpy as np
import warnings


# import scipy as sp


class SSECalculator:
    """
    Calculates the Sum of Squared Errors (SSE) for a given feature mask
    using NumPy.
    """

    def __init__(self, xtwx, xtwy, yty, regularization, rank=0):
        """
        Initializes the SSECalculator. All calculations are pre-staged
        using NumPy.

        Args:
            bases (np.ndarray): The basis matrix (X).
            energies (np.ndarray): The energy vector (y).
            counts (np.ndarray): The counts/weights vector (W).
        """

        self.xtwx = np.asarray(xtwx)
        self.xtwy = np.asarray(xtwy)
        self.yty = yty

        # Apply Tikihonov regularization
        self.xtwx = self.xtwx + regularization * np.eye(len(self.xtwy))

        self.base_sse = 1
        self.base_sse = self.calculate(np.ones_like(self.xtwy).astype(bool))

        cond = np.linalg.cond(self.xtwx)

        if cond > 1e10 and rank == 0:
            warnings.warn(
                f"Matrix is ill-conditioned: {cond:.2e} > 1e10 . Please consider more regularization."
            )

    def calculate(self, mask):
        """
        Calculates SSE for a given feature mask. This method is called
        in the worker process.

        Args:
            mask (np.ndarray): A boolean array indicating which features to include.

        Returns:
            float: The calculated Sum of Squared Errors.
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

        return (self.yty - theta @ xtwym) / self.base_sse
