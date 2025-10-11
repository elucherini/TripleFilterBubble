from dataclasses import dataclass, field
from typing import NewType, Callable
import numpy as np
from global_params import Params
from collections import defaultdict
import math

GuyId = NewType("GuyId", int)
InfobitId = NewType("InfobitId", int)


@dataclass(slots=True)
class Guy:
    id: GuyId
    position: np.ndarray
    group: int
    fluctuation: float = 0.0
    old_position: np.ndarray = field(init=False)
    inf_sum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    inf_count: int = 0

    def __post_init__(self):
        self.old_position = self.position.copy()

    @staticmethod
    def random_setup(guy_id: int, params: Params, rng: np.random.Generator) -> "Guy":
        group = int(rng.integers(0, params.numgroups))
        position = rng.uniform(-params.max_pxcor, params.max_pxcor, 2)
        return Guy(id=GuyId(guy_id), group=group, position=position)
    
    def update_fluctuation(self, max_pxcor: float, max_pycor: float):
        """
        fluctuation = distance_moved / (max_pxcor + 0.5) with torus wrapping

        max_pxcor, max_pycor: world size
        """
        half_x = float(max_pxcor) + 0.5
        p = self.position
        q = self.old_position

        half_y = (float(max_pycor) + 0.5)
        dx = float(p[0] - q[0])
        dy = float(p[1] - q[1])
        span_x = 2 * half_x
        span_y = 2 * half_y
        dx = ((dx + half_x) % span_x) - half_x
        dy = ((dy + half_y) % span_y) - half_y
        dist = math.hypot(dx, dy)

        self.fluctuation = dist / half_x
        self.old_position = p.copy()


@dataclass(slots=True)
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
    

@dataclass(slots=True)
class BiAdj:
    g2i: dict[GuyId, set[InfobitId]] = field(default_factory=lambda: defaultdict(set))
    i2g: dict[InfobitId, set[GuyId]] = field(default_factory=lambda: defaultdict(set))
    # Track who shared each infobit to each guy: (guy_id, infobit_id) -> sharer_guy_id
    sharer: dict[tuple[GuyId, InfobitId], GuyId] = field(default_factory=dict)
    # Optional callbacks for logging changes (wired by storage):
    _on_add: Callable[[GuyId, InfobitId], None] | None = None
    _on_remove: Callable[[GuyId, InfobitId], None] | None = None

    def add(self, gid: GuyId, iid: InfobitId):
        self.g2i[gid].add(iid)
        self.i2g[iid].add(gid)
        if self._on_add:
            self._on_add(gid, iid)

    def remove(self, gid: GuyId, iid: InfobitId):
        """Bidirectional edge removal."""
        if iid in self.g2i[gid]:
            self.g2i[gid].remove(iid)
            if self._on_remove:
                self._on_remove(gid, iid)
        if gid in self.i2g[iid]:
            self.i2g[iid].remove(gid)
        if not self.g2i[gid]:
            del self.g2i[gid]
        if not self.i2g[iid]:
            del self.i2g[iid]
        # Clean up sharer tracking
        self.sharer.pop((gid, iid), None)

    def neighbors_of_guy(self, gid: GuyId) -> set[InfobitId]:
        """Get all infobits connected to a guy."""
        return self.g2i.get(gid, set())

    def degree_of_info(self, iid: InfobitId) -> int:
        """Get the degree of an infobit."""
        return len(self.i2g.get(iid, set()))

    def drop_all_for_guy(self, gid: GuyId):
        # use list() to avoid mutating while iterating
        for iid in list(self.g2i.get(gid, ())):
            self.remove(gid, iid)

    def add_edge(self, gid: GuyId, iid: InfobitId) -> bool:
        s = self.g2i[gid]
        if iid in s:
            return False
        s.add(iid)
        self.i2g[iid].add(gid)
        if self._on_add:
            self._on_add(gid, iid)
        return True

    def remove_edge(self, gid: GuyId, iid: InfobitId) -> bool:
        if iid not in self.g2i.get(gid, ()):
            return False
        self.g2i[gid].remove(iid)
        self.i2g[iid].remove(gid)
        if not self.g2i[gid]:
            del self.g2i[gid]
        if not self.i2g[iid]:
            del self.i2g[iid]
        if self._on_remove:
            self._on_remove(gid, iid)
        # Clean up sharer tracking
        self.sharer.pop((gid, iid), None)
        return True