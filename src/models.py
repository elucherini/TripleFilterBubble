from dataclasses import dataclass, field
from typing import NewType
import numpy as np
from global_params import Params
from collections import defaultdict

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