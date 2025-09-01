import numpy as np

def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two sets of points.

    a: shape (N, d)
    b: shape (M, d)
    Returns: array of shape (N, M)
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    diff = a[:, None, :] - b[None, :, :]  # (N, M, d)
    return np.linalg.norm(diff, axis=2)   # (N, M)


def integration_probability(dist: float, lam: float, k: float) -> float:
    # lambda^k / (dist^k + lambda^k)
    # Note: in NL, dist is normalized by (max-pxcor + 0.5). We do that at call sites.
    if dist <= 0.0:
        return 1.0
    num = lam**k
    den = (dist**k + lam**k)
    return num / den
