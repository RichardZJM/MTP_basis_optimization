import numpy as np
import warnings
from typing import Tuple, Union


class SSECalculator:
    def __init__(
        self,
        tr_xtwx: np.ndarray,
        tr_xtwy: np.ndarray,
        tr_ytwy: float,
        val_xtwx: np.ndarray = None,
        val_xtwy: np.ndarray = None,
        val_ytwy: float = None,
        regularization: float = 0.0,
        rank: int = 0,
    ):
        # Training matrices
        self.tr_xtwx = np.asarray(tr_xtwx) + regularization * np.eye(len(tr_xtwy))
        self.tr_xtwy = np.asarray(tr_xtwy)
        self.tr_ytwy = np.asarray(tr_ytwy)

        # Validation matrices (optional)
        self.use_validation = val_xtwx is not None
        if self.use_validation:
            self.val_xtwx = np.asarray(val_xtwx)
            self.val_xtwy = np.asarray(val_xtwy)
            self.val_ytwy = np.asarray(val_ytwy)

        self.base_sse = 1
        self.base_sse = self.calculate(np.ones_like(self.tr_xtwy).astype(bool))

        if rank == 0:
            print(f"The base SSE is {self.base_sse:.2f}.")

        cond = np.linalg.cond(self.tr_xtwx)
        if cond > 1e10 and rank == 0:
            warnings.warn(
                f"Matrix is ill-conditioned: {cond:.2e} > 1e10 . Please consider more regularization."
            )

    def calculate(
        self, mask: np.ndarray, get_theta: bool = False
    ) -> Union[float, Tuple[np.ndarray, float]]:

        # 1. Fit theta on the masked TRAINING set
        xtwxm_tr = self.tr_xtwx[mask][:, mask]
        xtwym_tr = self.tr_xtwy[mask]

        try:
            theta = np.linalg.solve(xtwxm_tr, xtwym_tr)
        except np.linalg.LinAlgError:
            return float("inf")

        # 2. Evaluate SSE
        if self.use_validation:
            # Mask the validation matrices
            xtwxm_val = self.val_xtwx[mask][:, mask]
            xtwym_val = self.val_xtwy[mask]

            # Full quadratic expansion: yTy - 2*(theta^T XTy) + theta^T XTX theta
            term1 = self.val_ytwy
            term2 = 2 * np.dot(theta, xtwym_val)
            term3 = np.dot(theta, np.dot(xtwxm_val, theta))  # theta.T @ XTX @ theta

            sse = (term1 - term2 + term3) / self.base_sse
        else:
            # Training shortcut: yTy - theta^T XTy
            sse = (self.tr_ytwy - np.dot(theta, xtwym_tr)) / self.base_sse

        if get_theta:
            return (theta, sse.item())
        return sse.item()
