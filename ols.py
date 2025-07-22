import numpy as np

energies = np.genfromtxt("data/energies.txt", delimiter=",")
bases = np.genfromtxt("data/bases.txt", delimiter=" ")
counts = np.genfromtxt("data/counts.txt", delimiter=",")


XTX = bases.T @ bases
XTy = bases.T @ energies


def calculate_sse(mask):
    """Calculates sum of squared errors for a given feature mask."""
    XTXm = XTX[mask][:, mask]
    XTym = XTy[mask]
    theta = np.linalg.solve(XTXm, XTym)
    errs = bases[:, mask] @ theta - energies  # * counts
    return (np.sum(errs**2) / len(errs)) ** 0.5


if __name__ == "__main__":
    mask = np.ones(164).astype(bool)
    sse = calculate_sse(mask)
    print(sse)

    mask[12] = False
    sse = calculate_sse(mask)
    print(sse)
