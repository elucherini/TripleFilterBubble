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


class FastGeo:
    def __init__(self, max_pxcor: float, lam: float, k: float):
        self.inv_norm = 1.0 / (max_pxcor + 0.5)
        self.k = k
        self.k_half = 0.5 * k
        self.inv_norm_pow_k = self.inv_norm ** k
        self.lam_pow_k = lam ** k

    @staticmethod
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx*dx + dy*dy

    def integration_prob_from_d2(self, d2: float) -> float:
        # uses (sqrt(d2)*inv_norm)^k = (d2^(k/2)) * inv_norm^k
        x = (d2 ** self.k_half) * self.inv_norm_pow_k
        return self.lam_pow_k / (x + self.lam_pow_k)