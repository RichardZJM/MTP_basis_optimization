import cupy as cp
import numpy as np

# --- 1. One-Time Data Loading and GPU Transfer ---

energies_np = np.genfromtxt("data/energies.txt", delimiter=",")
bases_np = np.genfromtxt("data/bases.txt", delimiter=" ")
counts_np = np.genfromtxt("data/counts.txt", delimiter=",")

energies_gpu = cp.asarray(energies_np)
bases_gpu = cp.asarray(bases_np)
counts_gpu = cp.asarray(counts_np)

XTW = bases_gpu.T * counts_gpu
XTWX_gpu = XTW @ bases_gpu
XTWy_gpu = XTW @ energies_gpu


# --- 2. The GPU-Accelerated SSE Function ---


def calculate_sse_gpu(mask):
    """
    Calculates sum of squared errors on the GPU for a given feature mask.
    """
    mask_gpu = cp.asarray(mask)
    XTXm = XTWX_gpu[mask_gpu][:, mask_gpu]
    XTym = XTWy_gpu[mask_gpu]

    try:
        theta = cp.linalg.solve(XTXm, XTym)
    except cp.linalg.LinAlgError:
        return float("inf")

    residuals = (bases_gpu[:, mask_gpu] @ theta - energies_gpu) * counts_gpu
    sse = cp.sum(residuals**2)
    return sse.get()
