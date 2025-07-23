import numpy as np

energies = np.genfromtxt("data/energies.txt", delimiter=",")
bases = np.genfromtxt("data/bases.txt", delimiter=" ")
counts = np.genfromtxt("data/counts.txt", delimiter=",")

XTW = bases.T * counts
XTWX = XTW @ bases
XTWy = XTW @ energies


def calculate_sse(mask):
    """Calculates sum of squared errors for a given feature mask."""
    XTWXm = XTWX[mask][:, mask]
    XTWym = XTWy[mask]
    theta = np.linalg.solve(XTWXm, XTWym)
    residuals = (bases[:, mask] @ theta - energies) * counts
    return np.sum(residuals**2)


if __name__ == "__main__":
    mask = np.ones(164).astype(bool)
    sse = calculate_sse(mask)
    print(sse)

    mask[12] = False
    sse = calculate_sse(mask)
    print(sse)
