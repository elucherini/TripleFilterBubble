from global_params import Params
from models import Guy, GuyId, Infobit, InfobitId, BiAdj
import networkx as nx
import cProfile
import numpy as np
from dataclasses import dataclass, field
from utils import FastGeo
import itertools
import time


@dataclass
class Simulation:
    guys: dict[GuyId, Guy]
    G: nx.Graph
    H: BiAdj
    rng: np.random.Generator
    params: Params
    geo: FastGeo
    infobits: dict[InfobitId, Infobit] = field(default_factory=dict)

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

        return Simulation(guys=guys, G=G, H=H, rng=rng, params=params, geo=geo)

    def new_infobits(self, params: Params):
        # Store current positions as old positions
        for g in self.guys.values():
            g.old_position = g.position.copy()
        if params.new_info_mode == "central":
            for i in range(params.numcentral):
                new_central_infobit = Infobit.random_setup(len(self.infobits), params, self.rng)
                self.infobits[new_central_infobit.id] = new_central_infobit
                for guy in self.guys.values():
                    self.try_integrate_infobit(guy, new_central_infobit, params)
        elif params.new_info_mode == "individual":
            # The difference here is that each guy creates one infobit and tries to integrate it
            # as opposed to one infobit shared by all guys
            for guy in self.guys.values():
                new_individual_infobit = Infobit.random_setup(len(self.infobits), params, self.rng)
                self.infobits[new_individual_infobit.id] = new_individual_infobit
                self.try_integrate_infobit(guy, new_individual_infobit, params)
        elif params.new_info_mode in ("select close infobits", "select distant infobits"):
            is_close = params.new_info_mode == "select close infobits"
            for guy in self.guys.values():
                # Find all infobits that are not linked and fit closeness criteria
                candidate_infobits = []
                linked_infobits = self.infolink_neighbors(guy)
                for infobit_id in self.infobits:
                    if infobit_id in linked_infobits:
                        continue
                    distance = self.geo.dist2(guy.position, self.infobits[infobit_id].position)
                    if (is_close and distance < params.acceptance_latitude) or ((not is_close) and distance >= params.acceptance_latitude):
                        candidate_infobits.append(infobit_id)
                if not candidate_infobits or len(candidate_infobits) < len(self.guys):
                    new_infobit = Infobit.random_setup(len(self.infobits), params, self.rng)
                    self.infobits[new_infobit.id] = new_infobit
                    self.try_integrate_infobit(guy, new_infobit, params)
                else:
                    random_infobit_id = InfobitId(int(self.rng.choice(candidate_infobits)))
                    self.try_integrate_infobit(guy, self.infobits[random_infobit_id], params)
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
        guy_position = guy.position
        # For now, distance will be a float, but may change as we vectorize more
        d2 = self.geo.dist2(guy_position, infobit.position)
        integration_prob = self.geo.integration_prob_from_d2(d2)
        if self.rng.random() < integration_prob:
            # Integrate infobit in H graph passed down from Model
            # memory cap: if count my-infolinks >= memory, drop one infolink
            infolink_neighbors = self.infolink_neighbors(guy)
            if len(infolink_neighbors) >= params.memory:
                # drop a random infolink
                drop_iid = InfobitId(self.rng.choice(list(infolink_neighbors)))
                self.H.remove(guy.id, drop_iid)
            # add infolink
            self.H.add(guy.id, infobit.id)
            # set guy position to mean of infolink neighbors positions
            infos = self.infolink_neighbors(guy)
            if infos:
                xs = [self.infobits[i].position[0] for i in infos]
                ys = [self.infobits[i].position[1] for i in infos]
                guy.position[0] = float(np.mean(xs))
                guy.position[1] = float(np.mean(ys))

    def post_infobits(self, params: Params):
        for guy in self.guys.values():
            infos = self.infolink_neighbors(guy)
            if not infos:
                continue
            posted_info_id = InfobitId(self.rng.choice(list(infos)))
            neighbor_ids = list(self.G.neighbors(guy.id))
            for friend_id in neighbor_ids:
                self.try_integrate_infobit(self.guys[GuyId(friend_id)], self.infobits[posted_info_id], params)

    def birth_death(self, params: Params):
        """
        Replace a guy with a new guy with the same ID as a way to delete the old guy and reinsert a new guy.
        """
        for gid, guy in self.guys.items():
            if self.rng.random() < params.birth_death_probability:
                new_guy = Guy.random_setup(gid, params, self.rng)
                self.guys[gid] = new_guy
                # Remove all infolinks from the graph (they were from the old guy)
                for infolink in self.infolink_neighbors(guy):
                    self.H.remove(guy.id, infolink)

    def refriend(self, params: Params):
        if params.refriend_probability <= 0:
            return

        G = self.G
        edges = list(G.edges())  # avoid live view mutation cost

        # cache neighbors as sets (once per tick)
        neigh = {n: set(G.neighbors(n)) for n in G.nodes()}

        for (g1, g2) in edges:
            a = self.guys[GuyId(g1)].position
            b = self.guys[GuyId(g2)].position
            d2 = self.geo.dist2(a, b)
            dislike = 1.0 - self.geo.integration_prob_from_d2(d2)

            if self.rng.random() < params.refriend_probability * dislike:
                me = int(self.rng.choice([g1, g2]))
                # friends-of-friends not including self or current friends
                fof = set().union(*(neigh[n] for n in neigh[me])) - neigh[me] - {me}

                if fof:
                    new_friend = int(self.rng.choice(list(fof)))
                else:
                    # fallback: any non-friend
                    candidates = [x for x in G.nodes() if x != me and x not in neigh[me]]
                    new_friend = int(self.rng.choice(candidates)) if candidates else None

                if new_friend is not None:
                    G.add_edge(me, new_friend)
                    # keep caches coherent (cheap set updates)
                    neigh[me].add(new_friend)
                    neigh[new_friend].add(me)
                    if G.has_edge(g1, g2):
                        G.remove_edge(g1, g2)
                        neigh[g1].discard(g2)
                        neigh[g2].discard(g1)


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


    def run(self):
        for tick in range(self.params.numticks):
            start_time = time.time()
            self.new_infobits(self.params)
            if self.params.posting:
                self.post_infobits(self.params)
            if self.params.birth_death_probability > 0:
                self.birth_death(self.params)
            if self.params.refriend_probability > 0:
                self.refriend(self.params)
            self.update_infobits()
            print(f"Time taken for tick {tick}: {time.time() - start_time} seconds")
            # visualize(self.params)
            # if tick % self.params.plot_update_every == 0:
            #     create_infosharer_network(self.params)

def main():
    profiler = cProfile.Profile()
    profiler.enable()
    params = Params()
    model = Simulation.from_params(params)
    model.run()
    profiler.disable()
    profiler.dump_stats("optimized.prof")



if __name__ == "__main__":
    main()
