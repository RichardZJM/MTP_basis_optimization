import cupy as cp
import numpy as np

# --- 1. One-Time Data Loading and GPU Transfer ---

energies_np = np.genfromtxt("data/energies.txt", delimiter=",")
bases_np = np.genfromtxt("data/bases.txt", delimiter=" ")
# counts_np = np.genfromtxt("data/counts.txt", delimiter=",") # Uncomment if you use counts

energies_gpu = cp.asarray(energies_np)
bases_gpu = cp.asarray(bases_np)
# counts_gpu = cp.asarray(counts_np) # Uncomment if you use counts

XTX_gpu = bases_gpu.T @ bases_gpu
XTy_gpu = bases_gpu.T @ energies_gpu


# --- 2. The GPU-Accelerated SSE Function ---


def calculate_sse_gpu(mask):
    """
    Calculates sum of squared errors on the GPU for a given feature mask.
    """
    mask_gpu = cp.asarray(mask)
    XTXm = XTX_gpu[mask_gpu][:, mask_gpu]
    XTym = XTy_gpu[mask_gpu]

    try:
        theta = cp.linalg.solve(XTXm, XTym)
    except cp.linalg.LinAlgError:
        return float("inf")

    errs = bases_gpu[:, mask_gpu] @ theta - energies_gpu

    rmse = (cp.sum(errs**2) / len(errs)) ** 0.5
    return rmse.get()
