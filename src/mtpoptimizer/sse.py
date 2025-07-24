import numpy as np


class SSECalculator:
    """
    Calculates the Sum of Squared Errors (SSE) for a given feature mask
    using NumPy (CPU).
    """

    def __init__(self, bases, energies, counts):
        """
        Initializes the SSECalculator. All calculations are pre-staged
        using NumPy.

        Args:
            bases (np.ndarray): The basis matrix (X).
            energies (np.ndarray): The energy vector (y).
            counts (np.ndarray): The counts/weights vector (W).
        """

        self.bases = np.asarray(bases)
        self.energies = np.asarray(energies)
        self.counts = np.asarray(counts)

        xtw = self.bases.T * self.counts
        self.xtwx = xtw @ self.bases
        self.xtwy = xtw @ self.energies

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
        except np.linalg.LinAlgError:
            # If the matrix is singular, the system can't be solved.
            return float("inf")

        predicted_energies = self.bases[:, mask] @ theta
        residuals = (predicted_energies - self.energies) * self.counts
        sse = np.sum(residuals**2)

        return float(sse)
