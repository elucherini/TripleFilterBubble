from global_params import Params
from models import Guy, GuyId, Infobit, InfobitId, BiAdj
import networkx as nx
import cProfile
import numpy as np
from dataclasses import dataclass, field
from utils import FastGeo, FastStorage, SpatialGrid
import time
import pstats
import itertools
import pathlib
import zstandard as zstd


@dataclass
class Simulation:
    guys: dict[GuyId, Guy]
    G: nx.Graph
    H: BiAdj
    rng: np.random.Generator
    params: Params
    geo: FastGeo
    storage: FastStorage
    infobits: dict[InfobitId, Infobit] = field(default_factory=dict)
    _grid: SpatialGrid | None = None

    @staticmethod
    def make_group_network(guy: Guy, guys: dict[GuyId, Guy], G: nx.Graph, params: Params, rng: np.random.Generator):
        p = params.numfriends / max(1, params.numguys)  # baseline

        # current GID
        gid = guy.id
        if params.fraction_inter == 0:
            p_inter = 0.0
            p_intra = p * params.numgroups
        else:
            # Ensure expected total friend count aligns with numfriends
            p_inter = params.numgroups * p / ((1 - params.fraction_inter) / params.fraction_inter + params.numgroups - 1)
            p_intra = p_inter * (1 - params.fraction_inter) / params.fraction_inter

        my_group = guy.group
        for other_gid, other in guys.items():
            if other_gid == gid:
                continue
            # remove existing edge (parity with NL "if friend-neighbor? myself ask link-with myself [die]")
            # NOTE: I don't think this is needed, might be a NetLogo thing if you want to run multiple times
            if G.has_edge(gid, other.id):
                G.remove_edge(gid, other.id)
            # add with appropriate probability
            if other.group == my_group:
                if rng.random() < p_intra:
                    G.add_edge(gid, other.id)
            else:
                if rng.random() < p_inter:
                    G.add_edge(gid, other.id)

    @staticmethod
    def create_guys(params: Params, rng: np.random.Generator):
        return {GuyId(i): Guy.random_setup(i, params, rng) for i in range(params.numguys)}

    @staticmethod
    def from_params(params: Params):
        rng = np.random.default_rng(params.seed)
        # Create guys
        guys = Simulation.create_guys(params, rng)

        # Create group network for each guy
        G = nx.Graph()
        for curr_guy in guys.values():
            Simulation.make_group_network(curr_guy, guys, G, params, rng)

        # Create empty network of infolinks (connecting guys and infobits)
        H = BiAdj()
        geo = FastGeo(params.max_pxcor, params.acceptance_latitude, params.acceptance_sharpness)

        storage = FastStorage(params)

        if params.new_info_mode in ("select close infobits", "select distant infobits"):
            grid = SpatialGrid(params.max_pxcor, params.acceptance_latitude)
        else:
            grid = None

        return Simulation(guys=guys, G=G, H=H, rng=rng, params=params, geo=geo, storage=storage, _grid=grid)
    
    def _pick_distant_infobit_fast(self, guy: Guy, attempts: int = 32):
        """
        Pick a distant infobit for a guy with a random draw. 
        Tries as many times as specified by attempts, and returns None if no distant infobit is found.
        On average, this is still more efficient than scanning through all infobits.
        """
        if not self.infobits: return None
        linked = self.H.g2i.get(guy.id, ())
        keys = tuple(self.infobits.keys())
        for _ in range(attempts):
            iid = InfobitId(int(self.rng.choice(keys)))
            if iid in linked: continue
            if self.geo.norm_dist(guy.position, self.infobits[iid].position) >= self.params.acceptance_latitude:
                return self.infobits[iid]
        return None  # create new instead
    
    def _pick_close_infobit_from_grid(self, guy: Guy):
        if self._grid is None:
            return None
        linked = self.H.g2i.get(guy.id, ())
        # Scan only the 3x3 grid around the guy
        for iid in self._grid.neighbors(guy.position):
            if iid in linked:
                continue
            infobit = self.infobits[iid]
            if self.geo.norm_dist(guy.position, infobit.position) < self.params.acceptance_latitude:
                return infobit
        return None

    def new_infobits(self, params: Params):
        # Store current positions as old positions
        for g in self.guys.values():
            g.old_position = g.position.copy()
        if params.new_info_mode == "central":
            for i in range(params.numcentral):
                new_central_infobit = Infobit.random_setup(len(self.infobits), params, self.rng)
                self.infobits[new_central_infobit.id] = new_central_infobit
                if self._grid is not None:
                    self._grid.add(new_central_infobit.id, new_central_infobit.position)
                for guy in self.guys.values():
                    self.try_integrate_infobit(guy, new_central_infobit, params)
        elif params.new_info_mode == "individual":
            # The difference here is that each guy creates one infobit and tries to integrate it
            # as opposed to one infobit shared by all guys
            for guy in self.guys.values():
                new_individual_infobit = Infobit.random_setup(len(self.infobits), params, self.rng)
                self.infobits[new_individual_infobit.id] = new_individual_infobit
                if self._grid is not None:
                    self._grid.add(new_individual_infobit.id, new_individual_infobit.position)
                self.try_integrate_infobit(guy, new_individual_infobit, params)
        elif params.new_info_mode in ("select close infobits", "select distant infobits"):
            is_close = params.new_info_mode == "select close infobits"
            for guy in self.guys.values():
                if not is_close:
                    infobit = self._pick_distant_infobit_fast(guy)
                    if infobit is None:
                        infobit = Infobit.random_setup(len(self.infobits), params, self.rng)
                        self.infobits[infobit.id] = infobit
                        if self._grid is not None:
                            self._grid.add(infobit.id, infobit.position)
                    self.try_integrate_infobit(guy, infobit, params)
                else:
                    infobit = self._pick_close_infobit_from_grid(guy)
                    if infobit is None:
                        infobit = Infobit.random_setup(len(self.infobits), params, self.rng)
                        self.infobits[infobit.id] = infobit
                        if self._grid is not None:
                            self._grid.add(infobit.id, infobit.position)
                    self.try_integrate_infobit(guy, infobit, params)

        else:
            raise ValueError(f"Not yet implemented new_info_mode: {params.new_info_mode}")

    def infolink_neighbors(self, guy: Guy) -> set[InfobitId]:
        # Get all infolinks connected to guy
        return self.H.neighbors_of_guy(guy.id)

    def try_integrate_infobit(self, guy: Guy, infobit: Infobit, params: Params):
        """ If the infobit is integrated, this updates the guy's position to the mean of 
        the infobit's neighbors' positions. If the infobit is not integrated or it is already
        integrated, it does nothing.
        """
        linked = self.H.g2i[guy.id]
        if infobit.id in linked:
            return

        d2 = self.geo.dist2(guy.position, infobit.position)
        if self.rng.random() >= self.geo.integration_prob_from_d2(d2):
            return

        # memory cap: drop one
        if len(linked) >= params.memory:
            drop_iid = InfobitId(self.rng.choice(tuple(linked)))
            if self.H.remove_edge(guy.id, drop_iid):
                # decrement sums
                p = self.infobits[drop_iid].position
                guy.inf_sum -= p
                guy.inf_count -= 1

        # add new link and update sums
        if self.H.add_edge(guy.id, infobit.id):
            guy.inf_sum += infobit.position
            guy.inf_count += 1
            # new mean without allocations / reductions
            guy.position[:] = (guy.inf_sum / guy.inf_count)

    def _accept_mask_from_d2(self, d2_array: np.ndarray) -> np.ndarray:
        # p = lam^k / ( (sqrt(d2)*inv_norm)^k + lam^k )
        # → use (d2^(k/2)) * inv_norm^k
        x = np.power(d2_array, self.geo.k_half) * self.geo.inv_norm_pow_k
        p = self.geo.lam_pow_k / (x + self.geo.lam_pow_k)
        return self.rng.random(size=p.size) < p

    def post_infobits(self, params: Params):
        for guy in self.guys.values():
            infos = self.infolink_neighbors(guy)
            if not infos:
                continue
            k = int(self.rng.integers(len(infos)))
            posted_info_id = next(itertools.islice(infos, k, None))
            info = self.infobits[posted_info_id]
            for friend_id in self.G.adj[guy.id].keys():
                self.try_integrate_infobit(self.guys[GuyId(friend_id)], info, params)

    def birth_death(self, params: Params):
        for gid, guy in self.guys.items():
            if self.rng.random() < params.birth_death_probability:
                # drop all infolinks + fix sums
                for iid in tuple(self.H.g2i.get(gid, ())):
                    if self.H.remove_edge(gid, iid):
                        p = self.infobits[iid].position
                        guy.inf_sum -= p
                        guy.inf_count -= 1
                new_guy = Guy.random_setup(gid, params, self.rng)
                self.guys[gid] = new_guy

    def refriend(self, params: Params):
        G = self.G
        edges = list(G.edges())
        neigh = {n: set(G.neighbors(n)) for n in G.nodes()}
        for (g1, g2) in edges:
            a = self.guys[GuyId(g1)].position
            b = self.guys[GuyId(g2)].position
            d2 = self.geo.dist2(a, b)
            dislike = 1.0 - self.geo.integration_prob_from_d2(d2)
            if self.rng.random() >= params.refriend_probability * dislike:
                continue

            me = int(self.rng.choice([g1, g2]))
            fof = set()
            for n in neigh[me]:
                fof.update(neigh[n])
            fof.difference_update(neigh[me])
            fof.discard(me)

            if fof:
                new_friend = int(self.rng.integers(0, len(fof)))
                new_friend = list(fof)[new_friend]  # avoid full Python RNG on list
            else:
                # fallback
                cand = [x for x in G.nodes() if x != me and x not in neigh[me]]
                new_friend = int(self.rng.choice(cand)) if cand else None

            if new_friend is not None:
                if not G.has_edge(me, new_friend):
                    G.add_edge(me, new_friend)
                    neigh[me].add(new_friend); neigh[new_friend].add(me)
                if G.has_edge(g1, g2):
                    G.remove_edge(g1, g2)
                    neigh[g1].discard(g2); neigh[g2].discard(g1)

    def update_infobits(self):
        indices_to_remove = []
        for infobit_id in self.infobits:
            deg = self.H.degree_of_info(infobit_id)
            if deg == 0:
                # If degree is 0, the infobit is just not connected to any guys; so no need to remove it from H
                # We do need to remove it from the infobits dict
                indices_to_remove.append(infobit_id)
                continue
            self.infobits[infobit_id].popularity = deg
        for infobit_id in indices_to_remove:
            self.infobits.pop(infobit_id)

    def visualize(self):
        """Called this because it is called this in the NetLogo model, for us it's just updating the fluctuation and storing the results"""
        for guy in self.guys.values():
            guy.update_fluctuation(self.params.max_pxcor, self.params.max_pxcor)


    def run(self):
        self.storage.setup_writers(self.guys, self.params.numticks)
        if self.params.refriend_probability == 0:
            self.storage.precompute_guy_graph_row(self.G)
        self.storage.attach_biadj_callbacks(self.H)
        for tick in range(self.params.numticks):
            start_time = time.time()
            self.storage.begin_tick(tick)
            self.new_infobits(self.params)
            if self.params.posting:
                self.post_infobits(self.params)
            if self.params.birth_death_probability > 0:
                self.birth_death(self.params)
            if self.params.refriend_probability > 0:
                self.refriend(self.params)
            self.update_infobits()
            print(f"Time taken for tick {tick}: {time.time() - start_time} seconds")
            self.visualize()
            self.storage.write_tick(tick, self.guys)
            self.storage.write_guy_graph(tick, self.G)
            self.storage.end_tick(tick)
        self.storage.finalize(self.infobits)

def main():
    stats_name = "grid"
    profiler = cProfile.Profile()
    profiler.enable()
    params = Params()
    model = Simulation.from_params(params)
    model.run()
    profiler.disable()
    profiler.dump_stats(f"{stats_name}.prof")

    pstats.Stats(f"{stats_name}.prof").sort_stats('tottime').print_stats(30)



if __name__ == "__main__":
    main()
