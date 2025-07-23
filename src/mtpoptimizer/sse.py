# mtpoptimizer/sse.py
import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class SSECalculator:
    """
    Calculates the Sum of Squared Errors (SSE) for a given feature mask.
    cpu (numpy) or gpu (cupy)
    """

    def __init__(self, bases, energies, counts, device="cpu"):
        """
        Initializes the SSECalculator.

        Args:
            bases (np.ndarray): The basis matrix (X).
            energies (np.ndarray): The energy vector (y).
            counts (np.ndarray): The counts/weights vector (W).
            device (str): The compute device, 'cpu' or 'gpu'.
        """
        self.device = device

        xp = np
        if self.device == "gpu":
            if CUPY_AVAILABLE:
                xp = cp
            else:
                print("Warning: CuPy not found. Falling back to CPU.")
                self.device = "cpu"

        self.energies_d = xp.asarray(energies)
        self.bases_d = xp.asarray(bases)
        self.counts_d = xp.asarray(counts)

        xtw = self.bases_d.T * self.counts_d
        self.xtwx_d = xtw @ self.bases_d
        self.xtwy_d = xtw @ self.energies_d

    def calculate(self, mask):
        """
        Calculates SSE for a given feature mask. This method is called in the worker process.

        Args:
            mask (np.ndarray): A boolean array indicating which features to include.

        Returns:
            float: The calculated Sum of Squared Errors.
        """
        xp = np
        if self.device == "gpu" and CUPY_AVAILABLE:
            xp = cp

        mask_d = xp.asarray(mask)
        xtwxm = self.xtwx_d[mask_d][:, mask_d]
        xtwym = self.xtwy_d[mask_d]

        try:
            theta = xp.linalg.solve(xtwxm, xtwym)
        except xp.linalg.LinAlgError:
            return float("inf")  # Return infinity for singular matrices

        residuals = (self.bases_d[:, mask_d] @ theta - self.energies_d) * self.counts_d
        sse = xp.sum(residuals**2)

        if self.device == "gpu" and CUPY_AVAILABLE:
            return sse.get()
        return float(sse)
