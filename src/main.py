from global_params import Params
from models import Guy, GuyId, Infobit, InfobitId, BiAdj
import networkx as nx
import cProfile
import numpy as np
from dataclasses import dataclass, field
from utils import FastGeo, FastStorage, SpatialGrid
from plotter import PositionPlotter
from metrics import MeasurementResults, compute_metrics
import time
import pstats
import itertools


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
    infobits_created: int = 0
    plotter: PositionPlotter | None = None
    measurements: MeasurementResults | None = None

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
    def from_params(params: Params, enable_plotting: bool = False):
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

        plotter = PositionPlotter(params) if enable_plotting else None

        # Initialize measurements if any measurement ticks are configured
        measurements = MeasurementResults() if params.measurement_ticks else None

        return Simulation(guys=guys, G=G, H=H, rng=rng, params=params, geo=geo, storage=storage, _grid=grid, plotter=plotter, measurements=measurements)
    
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
                self.infobits_created += 1
                if self._grid is not None:
                    self._grid.add(new_central_infobit.id, new_central_infobit.position)
                GIDS = list(self.guys.keys())
                P = np.vstack([self.guys[gid].position for gid in GIDS])           # (N,2)
                d = P - new_central_infobit.position                                # (N,2)
                d2 = (d * d).sum(axis=1)                                            # (N,)
                mask = self._accept_mask_from_d2(d2)
                for gid in np.asarray(GIDS, dtype=int)[mask]:
                    self.try_integrate_infobit(self.guys[GuyId(gid)], new_central_infobit, params)
        elif params.new_info_mode == "individual":
            # The difference here is that each guy creates one infobit and tries to integrate it
            # as opposed to one infobit shared by all guys
            for guy in self.guys.values():
                new_individual_infobit = Infobit.random_setup(len(self.infobits), params, self.rng)
                self.infobits[new_individual_infobit.id] = new_individual_infobit
                self.infobits_created += 1
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
                        self.infobits_created += 1
                        if self._grid is not None:
                            self._grid.add(infobit.id, infobit.position)
                    self.try_integrate_infobit(guy, infobit, params)
                else:
                    infobit = self._pick_close_infobit_from_grid(guy)
                    if infobit is None:
                        infobit = Infobit.random_setup(len(self.infobits), params, self.rng)
                        self.infobits[infobit.id] = infobit
                        self.infobits_created += 1
                        if self._grid is not None:
                            self._grid.add(infobit.id, infobit.position)
                    self.try_integrate_infobit(guy, infobit, params)

        else:
            raise ValueError(f"Not yet implemented new_info_mode: {params.new_info_mode}")

    def infolink_neighbors(self, guy: Guy) -> set[InfobitId]:
        # Get all infolinks connected to guy
        return self.H.neighbors_of_guy(guy.id)

    def try_integrate_infobit(self, guy: Guy, infobit: Infobit, params: Params, sharer: Guy | None = None):
        """ If the infobit is integrated, this updates the guy's position to the mean of
        the infobit's neighbors' positions. If the infobit is not integrated or it is already
        integrated, it does nothing.

        Args:
            guy: The guy attempting to integrate the infobit
            infobit: The infobit to integrate
            params: Simulation parameters
            sharer: The guy who shared this infobit (None means self-created)
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
            # Track who shared this infobit (self if None)
            self.H.sharer[(guy.id, infobit.id)] = sharer.id if sharer else guy.id

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
            friends = list(self.G.adj[guy.id].keys())
            if not friends:
                continue
            F = np.vstack([self.guys[GuyId(fid)].position for fid in friends])
            diff = F - info.position
            d2 = (diff * diff).sum(axis=1)
            mask = self._accept_mask_from_d2(d2)
            for friend_id in np.asarray(friends, dtype=int)[mask]:
                self.try_integrate_infobit(self.guys[GuyId(friend_id)], info, params, sharer=guy)

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

    def plot_current_positions(
        self,
        title: str | None = None,
        color_by_group: bool = True,
        show_ids: bool = False,
        save_path: str | None = None,
        show_infolinks: bool = False,
        show_friend_links: bool = False
    ):
        """
        Plot current positions of all agents using the integrated plotter.

        Infobits are plotted if params.show_infobits is True.
        Infolinks are plotted if show_infolinks is True.
        Friend links are plotted if show_friend_links is True.

        Args:
            title: Plot title (defaults to tick-based title if None)
            color_by_group: If True, color agents by their group membership
            show_ids: If True, annotate each agent with their ID
            save_path: If provided, save the plot to this path instead of showing
            show_infolinks: If True, draw lines connecting guys to their infobits
            show_friend_links: If True, draw lines connecting guys to their friends

        Raises:
            RuntimeError: If plotter is not enabled for this simulation
        """
        if self.plotter is None:
            raise RuntimeError(
                "Plotter is not enabled for this simulation. "
                "Create simulation with enable_plotting=True"
            )

        if title is None:
            title = "Agent Positions in Opinion Space"

        self.plotter.plot_positions(
            guys=self.guys,
            title=title,
            color_by_group=color_by_group,
            show_ids=show_ids,
            save_path=save_path,
            infobits=self.infobits if self.params.show_infobits else None,
            show_infobits=self.params.show_infobits,
            size_by_popularity=self.params.infobit_size,
            H=self.H if show_infolinks else None,
            show_infolinks=show_infolinks,
            G=self.G if show_friend_links else None,
            show_friend_links=show_friend_links
        )


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

            # Compute measurements if this tick is configured for measurement
            if self.measurements is not None and tick in self.params.measurement_ticks:
                metrics = compute_metrics(self, tick)
                self.measurements.add_measurement(
                    tick=tick,
                    mean_link_length=metrics['mean_link_length'],
                    mean_infosharer_distance=metrics['mean_infosharer_distance'],
                    mean_friend_distance=metrics['mean_friend_distance']
                )
                ll_str = f"{metrics['mean_link_length']:.4f}" if metrics['mean_link_length'] is not None else 'N/A'
                isd_str = f"{metrics['mean_infosharer_distance']:.4f}" if metrics['mean_infosharer_distance'] is not None else 'N/A'
                fd_str = f"{metrics['mean_friend_distance']:.4f}" if metrics['mean_friend_distance'] is not None else 'N/A'
                print(f"[Tick {tick}] Measurements computed: "
                      f"link_length={ll_str}, infosharer_dist={isd_str}, friend_dist={fd_str}")

            print(f"Time taken for tick {tick}: {time.time() - start_time} seconds")
            self.visualize()

            # Plot positions if enabled
            if self.params.plot_every_n_ticks > 0:
                if tick % self.params.plot_every_n_ticks == 0 or tick == self.params.numticks - 1:
                    print(f"Plotting positions at tick {tick}...")
                    self.plot_current_positions(
                        title=f"Agent Positions at Tick {tick}",
                        color_by_group=True,
                        save_path=f"positions_tick_{tick:03d}.png",
                        show_infolinks=self.params.show_infolinks,
                        show_friend_links=self.params.show_friend_links
                    )

            self.storage.write_tick(tick, self.guys)
            self.storage.write_guy_graph(tick, self.G)
            self.storage.end_tick(tick)
        self.storage.finalize(self.infobits)

        # Print measurement summary at the end
        if self.measurements is not None:
            print("\n" + "="*60)
            print(self.measurements)
            print("="*60)

def main():
    stats_name = "posting"
    profiler = cProfile.Profile()
    profiler.enable()
    params = Params()
    enable_plotting = params.plot_every_n_ticks > 0
    model = Simulation.from_params(params, enable_plotting=enable_plotting)
    model.run()
    profiler.disable()
    profiler.dump_stats(f"{stats_name}.prof")

    pstats.Stats(f"{stats_name}.prof").sort_stats('tottime').print_stats(30)


if __name__ == "__main__":
    main()
