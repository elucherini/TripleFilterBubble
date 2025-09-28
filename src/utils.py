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