from dataclasses import dataclass, field
from typing import NewType
import numpy as np
from global_params import Params
from collections import defaultdict
import scipy.sparse as sp

GuyId = NewType("GuyId", int)
InfobitId = NewType("InfobitId", int)


@dataclass
class Guy:
    id: GuyId
    position: np.ndarray
    group: int
    fluctuation: float = 0.0
    old_position: np.ndarray = field(init=False)

    def __post_init__(self):
        self.old_position = self.position.copy()

    @staticmethod
    def random_setup(guy_id: int, params: Params, rng: np.random.Generator) -> "Guy":
        group = int(rng.integers(0, params.numgroups - 1))
        position = rng.uniform(-params.max_pxcor, params.max_pxcor, 2)
        return Guy(id=GuyId(guy_id), group=group, position=position)
    

@dataclass
class Guys:
    positions: np.ndarray  # (numguys, 2)
    old_positions: np.ndarray  # (numguys, 2)
    groups: np.ndarray  # (numguys,)
    fluctuations: np.ndarray  # (numguys,)
    ids: np.ndarray  # (numguys,)
    adj_matrix: sp.csr_array # (numguys, numguys)

    @staticmethod
    def make_group_network(params: Params, rng: np.random.Generator, groups: np.ndarray):
        p = params.numfriends / max(1, params.numguys - 1)

        if params.fraction_inter == 0:
            p_inter = 0.0
            p_intra = p * params.numgroups
        else:
            # Solve for p_inter/p_intra so expected total friends ≈ numfriends
            denom = ((1 - params.fraction_inter) / params.fraction_inter + params.numgroups - 1)
            p_inter = params.numgroups * p / denom
            p_intra = p_inter * (1 - params.fraction_inter) / params.fraction_inter
        
        # Probability matrix by group (upper triangle only)
        same = groups[:, None] == groups[None, :]
        P = np.where(same, p_intra, p_inter)

        # Sample only i<j, then mirror to keep graph symmetric and zero diagonal
        iu = np.triu_indices(params.numguys, k=1)
        mask_upper = (rng.random(iu[0].size) < P[iu])
        rows = iu[0][mask_upper]
        cols = iu[1][mask_upper]

        # Build sparse adjacency
        data = np.ones(rows.size * 2, dtype=np.uint8)
        adj = sp.csr_array((data, (np.concatenate([rows, cols]),
                                np.concatenate([cols, rows]))),
                        shape=(params.numguys, params.numguys))
        return adj

    @staticmethod
    def random_setup(params: Params, rng: np.random.Generator):
        N = params.numguys

        ids = np.arange(N, dtype=np.int32)
        groups = rng.integers(0, params.numgroups, size=N, dtype=np.int32)
        positions = rng.uniform(-params.max_pxcor, params.max_pxcor, size=(N, 2)).astype(np.float64)

        adj_matrix = Guys.make_group_network(params, rng, groups)

        old_positions = positions.copy()
        fluctuations = np.zeros(N, dtype=np.float64)

        return Guys(
            positions=positions,
            old_positions=old_positions,
            groups=groups,
            fluctuations=fluctuations,
            ids=ids,
            adj_matrix=adj_matrix
        )





@dataclass
class Infobit:
    id: InfobitId
    position: np.ndarray
    popularity: float = 0.0
    old_position: np.ndarray = field(init=False)

    def __post_init__(self):
        self.old_position = self.position.copy()

    @staticmethod
    def random_setup(infobit_id: int, params: Params, rng: np.random.Generator):
        position = rng.uniform(-params.max_pxcor, params.max_pxcor, 2)
        return Infobit(id=InfobitId(infobit_id), position=position)
    

@dataclass
class Infobits:
    positions: np.ndarray
    popularity: np.ndarray
    old_positions: np.ndarray = field(init=False)

    def __post_init__(self):
        self.old_positions = self.positions.copy()

    @staticmethod
    def empty():
        return Infobits(
            positions=np.empty((0, 2), dtype=np.float64),
            popularity=np.empty((0,), dtype=np.float64),
        )
    

@dataclass
class BiAdj:
    g2i: dict[GuyId, set[InfobitId]] = field(default_factory=lambda: defaultdict(set))
    i2g: dict[InfobitId, set[GuyId]] = field(default_factory=lambda: defaultdict(set))

    def add(self, gid: GuyId, iid: InfobitId):
        """Bidirectional edge addition."""
        self.g2i[gid].add(iid)
        self.i2g[iid].add(gid)

    def remove(self, gid: GuyId, iid: InfobitId):
        """Bidirectional edge removal."""
        if iid in self.g2i[gid]:
            self.g2i[gid].remove(iid)
        if gid in self.i2g[iid]:
            self.i2g[iid].remove(gid)
        if not self.g2i[gid]:
            del self.g2i[gid]
        if not self.i2g[iid]:
            del self.i2g[iid]

    def neighbors_of_guy(self, gid: GuyId) -> set[InfobitId]:
        """Get all infobits connected to a guy."""
        return self.g2i.get(gid, set())

    def degree_of_info(self, iid: InfobitId) -> int:
        """Get the degree of an infobit."""
        return len(self.i2g.get(iid, set()))

    def drop_all_for_guy(self, gid: GuyId):
        """Remove all infobits connected to a guy."""
        for iid in self.g2i.get(gid, set()):
            self.remove(gid, iid)


@dataclass
class Bipartite:
    """Row = guy (N), Col = infobit (M)"""
    H: sp.csr_array
    _Hc: sp.csc_array | None = field(init=False, default=None)

    @property
    def Hc(self) -> sp.csc_array:
        if self._Hc is None:
            self._Hc = self.H.tocsc()
        return self._Hc

    def invalidate_columns_view(self):
        self._Hc = None