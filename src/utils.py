import zstandard as zstd
from global_params import Params
import pathlib
import numpy as np
from pathlib import Path
from models import Guy, GuyId, Infobit, InfobitId, BiAdj
from typing import Iterable
import networkx as nx
from typing import BinaryIO


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
    def __init__(self, params: "Params"):
        self.params = params
        self.pos_mm = None
        self.run_dir = Path(params.run_dir)
        self.guy_ids: np.ndarray | None = None

        # positions + infobits (existing)
        self.guy_filename: str = "guy_positions_TxNx2_int16.npy"
        self.infobit_filename: str = "infobits_final_int16.npy"
        self.infobit_ids_filename: str = "infobit_ids.npy"

        # Guy↔Guy graph (upper-tri bitset per tick)
        self.gg_filename: str = "guy_graph_TxPackedBytes_uint8.npy"
        self.gg_mm: np.memmap | None = None
        self.gg_pair_index: np.ndarray | None = None  # (N,N)->bit idx
        self.gg_bytes_per_tick: int = 0
        self.gid2idx: dict[int, int] = {}

        # Bipartite events
        self.bi_events_filename: str = "biadj_events.bin"       # raw bytes: [gid(u2), iid(u4), op(u1)] *
        self.bi_counts_filename: str = "biadj_counts_uint32.npy" # T-length counts per tick
        self._bi_counts: np.ndarray | None = None    # np.uint32[T]
        self._bi_buf: list[tuple[int,int,int]] = []     # current tick buffer of (gid_idx,u2, iid_idx,u4, op 0/1)
        self._bi_fh: BinaryIO | None = None
        self._bi_open = False

        # Infobit ID indexer (stable order used in event log & final positions)
        self.inf_iid2idx: dict[int, int] = {}
        self.inf_ids_list: list[int] = []

        self._gg_static_row: np.ndarray | None = None

    def precompute_guy_graph_row(self, G: nx.Graph):
        buf = np.zeros((self.gg_bytes_per_tick,), np.uint8)
        for u, v in G.edges():
            iu, iv = self.gid2idx[int(u)], self.gid2idx[int(v)]
            if iu > iv: iu, iv = iv, iu
            k = self.gg_pair_index[iu, iv]
            buf[k >> 3] |= (1 << (k & 7))
        self._gg_static_row = buf

    def compress_fast(self, src: Path):
        dst = src.with_suffix(src.suffix + ".zst")
        cctx = zstd.ZstdCompressor(level=3, threads=0)
        with open(src, "rb") as f, open(dst, "wb") as g:
            cctx.copy_stream(f, g)

    def setup_writers(self, guys: dict[GuyId, Guy], T: int):
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Stable guy-order + indexer
        guy_ids = np.array([g.id for g in guys.values()], dtype=np.int64)
        order = np.argsort(guy_ids)
        self.guy_ids = guy_ids[order]
        self.gid2idx = {int(gid): i for i, gid in enumerate(self.guy_ids)}
        N = self.guy_ids.shape[0]

        # Positions memmap (existing)
        self.pos_mm = np.lib.format.open_memmap(
            self.run_dir / self.guy_filename, mode="w+", dtype=np.int16, shape=(T, N, 2)
        )

        # Graph: precompute (i,j)->bit index table and open bitset memmap
        M_bits = N * (N - 1) // 2
        self.gg_bytes_per_tick = (M_bits + 7) // 8
        self.gg_mm = np.lib.format.open_memmap(
            self.run_dir / self.gg_filename, mode="w+",
            dtype=np.uint8, shape=(T, self.gg_bytes_per_tick)
        )
        pair_index = np.full((N, N), -1, dtype=np.int32)
        k = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                pair_index[i, j] = k
                k += 1
        self.gg_pair_index = pair_index

        # Bipartite event writer + counts
        self._bi_fh = open(self.run_dir / self.bi_events_filename, "wb", buffering=1024*1024)
        self._bi_open = True
        self._bi_counts = np.zeros((T,), dtype=np.uint32)
        self._bi_buf = []

        # Save meta
        np.savez(self.run_dir / "meta.npz",
                 guy_ids=self.guy_ids,
                 max_pxcor=float(self.params.max_pxcor),
                 max_pycor=float(self.params.max_pxcor),
                 ticks=T,
                 num_guys=N,
                 gg_bytes_per_tick=self.gg_bytes_per_tick)

    def _quantize(self, position: np.ndarray) -> np.ndarray:
        if not self.params.quantize:
            return position
        scale = self.params.quantization_scale / (2 * self.params.max_pxcor)
        clip_by = int(self.params.quantization_scale)
        min_pos = -self.params.max_pxcor
        return np.rint((position - min_pos) * scale).clip(0, clip_by).astype(np.uint16)

    def write_tick(self, t: int, guys: dict[GuyId, Guy]):
        if self.pos_mm is None or self.guy_ids is None:
            raise ValueError("pos_mm is not set")
        row = self.pos_mm[t]
        for i, gid in enumerate(self.guy_ids):
            g = guys[int(gid)]
            row[i] = self._quantize(g.position)

    # -------- guy↔guy graph per tick --------
    def write_guy_graph(self, t: int, g: nx.Graph | Iterable[tuple[GuyId, GuyId]]):
        """
        Pack upper triangle (i<j) into bits; 1=edge present.
        Accepts a networkx.Graph (with GuyIds as nodes) OR an iterable of (u,v) edges.
        """
        if self.gg_mm is None or self.gg_pair_index is None:
            raise ValueError("graph buffers not set up")
        if self._gg_static_row is not None:
            self.gg_mm[t][:] = self._gg_static_row
            return
        
        buf = self.gg_mm[t]
        buf[:] = 0  # fast zero

        # Get an iterable of edges (GuyId, GuyId)
        edges = g.edges() if isinstance(g, nx.Graph) else g

        # Set bits for each edge
        pi = self.gg_pair_index
        for u, v in edges:
            iu = self.gid2idx.get(int(u))
            iv = self.gid2idx.get(int(v))
            if iu is None or iv is None or iu == iv:
                continue
            if iu > iv:
                iu, iv = iv, iu
            k = pi[iu, iv]
            if k < 0:
                continue
            b = k & 7
            by = k >> 3
            buf[by] |= (1 << b)

    # -------- bipartite event logging --------
    def begin_tick(self, t: int):
        # call once at the beginning of a tick; cheap
        self._bi_buf.clear()

    def bi_on_add(self, gid: GuyId, iid: InfobitId):
        gid_idx = self.gid2idx[int(gid)]
        iid_int = int(iid)
        # Map/assign infobit index on first sight
        if iid_int not in self.inf_iid2idx:
            self.inf_iid2idx[iid_int] = len(self.inf_ids_list)
            self.inf_ids_list.append(iid_int)
        iid_idx = self.inf_iid2idx[iid_int]
        self._bi_buf.append((gid_idx, iid_idx, 1))

    def bi_on_remove(self, gid: GuyId, iid: InfobitId):
        gid_idx = self.gid2idx[int(gid)]
        iid_int = int(iid)
        idx = self.inf_iid2idx.get(iid_int)
        if idx is None:
            return  # removing something we never logged → ignore
        self._bi_buf.append((gid_idx, idx, 0))

    def end_tick(self, t: int):
        # Flush this tick's events to disk as tightly packed bytes.
        if not self._bi_open:
            return
        cnt = len(self._bi_buf)
        self._bi_counts[t] = np.uint32(cnt)  # type: ignore
        if cnt == 0:
            return
        arr = np.empty(cnt, dtype=np.dtype([('gid','<u2'),('iid','<u4'),('op','u1')]))
        # fill
        arr['gid'] = [a for (a,_,_) in self._bi_buf]
        arr['iid'] = [b for (_,b,_) in self._bi_buf]
        arr['op']  = [c for (_,_,c) in self._bi_buf]
        self._bi_fh.write(arr.tobytes())  # type: ignore
        self._bi_buf.clear()

    # -------- housekeeping --------
    def _delete(self, file: Path):
        if file.exists():
            file.unlink()

    def _close_graph_buffers(self):
        if self.pos_mm is not None:
            self.pos_mm.flush()
        if self.gg_mm is not None:
            self.gg_mm.flush()
        if self._bi_open:
            self._bi_fh.flush()
            self._bi_fh.close()
            self._bi_open = False

    def finalize(self, infobits: dict[InfobitId, Infobit]):
        # close mmaps / files
        self._close_graph_buffers()

        # Persist bipartite counts
        np.save(self.run_dir / self.bi_counts_filename, self._bi_counts)  # type: ignore

        # Save infobit IDs + final positions in the SAME order used for events
        inf_ids = np.array(self.inf_ids_list, dtype=np.int64)
        inf_xy = np.empty((len(inf_ids), 2), dtype=np.int16)
        for i, iid in enumerate(inf_ids):
            ib = infobits[iid]
            inf_xy[i] = self._quantize(ib.position)
        np.save(self.run_dir / self.infobit_filename, inf_xy)  # type: ignore
        np.save(self.run_dir / self.infobit_ids_filename, inf_ids)

        # Compress everything heavy
        self.compress_fast(self.run_dir / self.guy_filename)
        self.compress_fast(self.run_dir / self.infobit_filename)
        self.compress_fast(self.run_dir / self.infobit_ids_filename)
        self.compress_fast(self.run_dir / self.gg_filename)
        self.compress_fast(self.run_dir / self.bi_events_filename)
        self.compress_fast(self.run_dir / self.bi_counts_filename)

        # Remove uncompressed
        self._delete(self.run_dir / self.guy_filename)
        self._delete(self.run_dir / self.infobit_filename)
        self._delete(self.run_dir / self.infobit_ids_filename)
        self._delete(self.run_dir / self.gg_filename)
        self._delete(self.run_dir / self.bi_events_filename)
        self._delete(self.run_dir / self.bi_counts_filename)

    # Convenience to attach to a BiAdj:
    def attach_biadj_callbacks(self, bi: BiAdj):
        bi._on_add = self.bi_on_add
        bi._on_remove = self.bi_on_remove