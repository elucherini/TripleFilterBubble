import zstandard as zstd
from global_params import Params
import pathlib
import numpy as np
from pathlib import Path


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
    

class FastStorage:
    def __init__(self, params: Params):
        self.params = params
        self.pos_mm = None
        # Order of guys
        self.guy_ids: np.ndarray | None = None
        self.run_dir: Path | None = None
        self.guy_filename: str = "guy_positions_TxNx2_int16.npy"
        self.infobit_filename: str = "infobits_final_int16.npy"
        self.infobit_ids_filename: str = "infobit_ids.npy"

    def compress_fast(self, src: Path):
        dst = src.with_suffix(src.suffix + ".zst")
        cctx = zstd.ZstdCompressor(level=3, threads=0)  # uses all cores
        with open(src, "rb") as f, open(dst, "wb") as g:
            cctx.copy_stream(f, g)

    def setup_writers(self, guys: dict["GuyId", "Guy"], T: int, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Fix a stable guy-order once
        guy_ids = np.array([g.id for g in guys.values()], dtype=np.int64)
        # (Optionally sort if you want deterministic ordering)
        order = np.argsort(guy_ids)
        self.guy_ids = guy_ids[order]

        N = guy_ids.shape[0]
        self.pos_mm = np.lib.format.open_memmap(
            self.run_dir / self.guy_filename,
            mode="w+",
            dtype=np.int16,
            shape=(T, N, 2),
        )

        # Save ID lists + any metadata you’ll need later
        np.savez(self.run_dir / "meta.npz",
                guy_ids=self.guy_ids,
                max_pxcor=float(self.params.max_pxcor),
                max_pycor=float(self.params.max_pxcor),
                ticks=T)

    def _quantize(self, position: np.ndarray) -> np.ndarray:
        if not self.params.quantize:
            return position
        scale = self.params.quantization_scale / (2 * self.params.max_pxcor)
        clip_by = int(self.params.quantization_scale)
        min_pos = -self.params.max_pxcor
        return np.rint((position - min_pos) * scale).clip(0, clip_by).astype(np.uint16)

    def write_tick(self, t: int, guys: dict["GuyId", "Guy"]):
        # Get a writable view of the row for this tick (no extra alloc)
        if self.pos_mm is None or self.guy_ids is None:
            raise ValueError("pos_mm is not set")
        row = self.pos_mm[t]

        # Fill it in the same order every tick (fast, contiguous)
        # If you sorted guy_ids, index sim.guys by id; no dict lookup if you keep a list
        for i, gid in enumerate(self.guy_ids):
            g = guys[gid]
            # assume you already have float positions; quantize to int16
            # clamp to bounds to avoid overflow just in case
            row[i] = self._quantize(g.position)
        # No flush here; let the OS buffer it

    def _delete(self, file: Path):
        if file.exists():
            file.unlink()

    def finalize(self, infobits):
        self.pos_mm.flush()  # ensure data hits disk

        # Infobits final positions
        inf_ids = np.array([ib.id for ib in infobits.values()], dtype=np.int64)
        inf_xy  = np.empty((len(inf_ids), 2), dtype=np.int16)
        for i, ib in enumerate(infobits.values()):
            inf_xy[i] = self._quantize(ib.position)
        np.save(self.run_dir / self.infobit_filename, inf_xy)
        np.save(self.run_dir / self.infobit_ids_filename, inf_ids)

        self.compress_fast(self.run_dir / self.guy_filename)
        self.compress_fast(self.run_dir / self.infobit_filename)
        self.compress_fast(self.run_dir / self.infobit_ids_filename)
        # Delete uncompressed files at the end
        self._delete(self.run_dir / self.guy_filename)
        self._delete(self.run_dir / self.infobit_filename)
        self._delete(self.run_dir / self.infobit_ids_filename)

