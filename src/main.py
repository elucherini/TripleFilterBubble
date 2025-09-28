from global_params import Params
from models import GuyId, Infobit, InfobitId, Guys, Bipartite, Infobits
import cProfile
import numpy as np
from dataclasses import dataclass
from utils import pairwise_distances, integration_probability
import itertools
import time
import scipy.sparse as sp


@dataclass
class Simulation:
    guys: Guys
    bipartite: Bipartite
    rng: np.random.Generator
    params: Params
    infobits: Infobits = Infobits.empty()

    @staticmethod
    def from_params(params: Params):
        rng = np.random.default_rng(params.seed)
        # Create guys
        guys = Guys.random_setup(params, rng)

        # Create empty network of infolinks (connecting guys and infobits)
        bipartite = Bipartite(H=sp.csr_array((params.numguys, 0)))

        return Simulation(guys=guys, bipartite=bipartite, rng=rng, params=params)

    def new_infobits(self, params: Params):
        if params.new_info_mode == "central":
            self.integrate_new_infobits_batch(params=params)
        elif params.new_info_mode == "individual":
            # The difference here is that each guy creates one infobit and tries to integrate it
            # as opposed to one infobit shared by all guys
            self.integrate_new_infobits_individual(params=params)
        elif params.new_info_mode in ("select close infobits", "select distant infobits"):
            is_close = params.new_info_mode == "select close infobits"
            for guy in self.guys.values():
                # Find all infobits that are not linked and fit closeness criteria
                candidate_infobits = []
                linked_infobits = self.infolink_neighbors(guy)
                for infobit_id in self.infobits:
                    if infobit_id in linked_infobits:
                        continue
                    distance = float(pairwise_distances(guy.position, self.infobits[infobit_id].position)) / (params.max_pxcor + 0.5)
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
    
    def integrate_new_infobits_batch(
        self,
        params: Params,
        chunk_cols: int = 2048
    ):
        B = self.params.numcentral  # number of new central infobits
        self.guys.old_positions = self.guys.positions.copy()
        N = self.guys.positions.shape[0]
        H = self.bipartite.H

        pos_g = self.guys.positions
        pos_i = self.infobits.positions
        M0 = pos_i.shape[0]

        # 1) create new infobit positions
        new_pos = self.rng.uniform(-params.max_pxcor, params.max_pxcor, size=(B, 2)).astype(np.float64)
        pos_i = np.vstack([pos_i, new_pos])          # shape now (M0+B, 2)

        lam = params.acceptance_latitude
        k   = params.acceptance_sharpness
        norm = params.max_pxcor + 0.5

        add_rows, add_cols = [], []
        rem_rows, rem_cols = [], []

        # degrees before this batch (row-wise)
        deg0 = np.asarray(H.sum(axis=1)).ravel().astype(int)

        # 2) chunked guy–new-infobit distances, probs, proposals
        for c0 in range(M0, M0 + B, chunk_cols):
            c1 = min(c0 + chunk_cols, M0 + B)
            C = c1 - c0

            diff = pos_g[:, None, :] - pos_i[c0:c1][None, :, :]  # (N, C, 2)
            D = np.linalg.norm(diff, axis=2) / norm              # (N, C)

            if k == 1:
                p = np.where(D <= 0.0, 1.0, lam / (D + lam))
            else:
                lamk = lam ** k
                Dk = np.where(D <= 0.0, 0.0, D ** k)
                p = np.where(D <= 0.0, 1.0, lamk / (Dk + lamk))

            props = self.rng.random(D.shape) < p                      # (N, C) bool
            if not props.any():
                continue

            # rows with at least one proposal in this chunk
            rows = np.nonzero(props.any(axis=1))[0]
            if rows.size == 0:
                continue

            # how many proposed per row in this chunk
            adds_count_chunk = props.sum(axis=1)                 # (N,)
            adds_count_rows = adds_count_chunk[rows]

            # collect proposed coordinates
            prop_cols_local = [np.nonzero(props[r])[0] for r in rows]
            prop_cols_abs   = [c0 + arr for arr in prop_cols_local]

            # 3) enforce memory with replacement:
            # need_drop = max(0, (deg0[g] + added_so_far[g] + adds_count_rows[g]) - memory)
            # track added_so_far per row to compute overflow accurately across chunks
            # Use a small dict only for touched rows to avoid a full-length array write.
            # (N is large; touched rows are typically small.)
            # Initialize lazily:
            if not hasattr(self.integrate_new_infobits_batch, "_added_so_far"):
                self.integrate_new_infobits_batch._added_so_far = {}
            added_so_far = self.integrate_new_infobits_batch._added_so_far

            for r, cols_abs in zip(rows, prop_cols_abs):
                t = cols_abs.size
                if t == 0:
                    continue
                a_prev = added_so_far.get(r, 0)
                overflow = deg0[r] + a_prev + t - params.memory
                need_drop = max(0, overflow)

                if need_drop > 0 and deg0[r] > 0:
                    # drop uniformly from current existing neighbors
                    i0, i1 = H.indptr[r], H.indptr[r+1]
                    existing = H.indices[i0:i1]
                    kdrop = min(need_drop, existing.size)
                    to_drop = self.rng.choice(existing, size=kdrop, replace=False)
                    rem_rows.extend([r] * kdrop)
                    rem_cols.extend(to_drop.tolist())
                    deg0[r] -= kdrop

                # add all proposals for this row (duplicates will be deduped later)
                add_rows.extend([r] * t)
                add_cols.extend(cols_abs.tolist())
                added_so_far[r] = a_prev + t

        # 4) apply removals and additions (boolean algebra to dedupe)
        if rem_rows:
            R = sp.csr_array(
                (np.ones(len(rem_rows), dtype=np.uint8),
                (np.array(rem_rows), np.array(rem_cols))),
                shape=H.shape
            )
            H = H - H.multiply(R)

        if add_rows:
            A = sp.csr_array(
                (np.ones(len(add_rows), dtype=np.uint8),
                (np.array(add_rows), np.array(add_cols))),
                shape=(N, pos_i.shape[0])
            )
            # H’s shape may have grown columns; align:
            if A.shape[1] != H.shape[1]:
                H = sp.hstack([H, sp.csr_array((N, A.shape[1] - H.shape[1]))]).tocsr()
            H = ((H + A) > 0).astype(np.uint8).tocsr()

        # invalidate CSC view if present
        self.bipartite.H = H
        self.bipartite.invalidate_columns_view()

        # 5) update guy positions = mean of all infolink neighbors
        deg = np.asarray(H.sum(axis=1)).ravel()
        nz = deg > 0
        if nz.any():
            sums = H @ pos_i   # (N,2)
            pos_g[nz] = sums[nz] / deg[nz, None]

        # clear per-call scratch
        if hasattr(self.integrate_new_infobits_batch, "_added_so_far"):
            self.integrate_new_infobits_batch._added_so_far = {}

        self.infobits.old_positions = self.infobits.positions.copy()
        self.infobits.positions = pos_i


    def integrate_new_infobits_individual(self, params: Params):
        self.guys.old_positions = self.guys.positions.copy()
        N = self.guys.positions.shape[0]
        H = self.bipartite.H

        pos_g = self.guys.positions
        pos_i = self.infobits.positions
        M0 = pos_i.shape[0]

        # 1) One new infobit per guy, uniformly sampled over the world
        new_pos = self.rng.uniform(-params.max_pxcor, params.max_pxcor, size=(N, 2)).astype(np.float64)
        pos_i = np.vstack([pos_i, new_pos])  # (M0+N, 2)

        # keep popularity aligned (new entries are 0)
        if hasattr(self.infobits, "popularity") and self.infobits.popularity is not None:
            new_pop = np.zeros((N,), dtype=self.infobits.popularity.dtype if self.infobits.popularity.size else np.float64)
            self.infobits.popularity = np.concatenate([self.infobits.popularity, new_pop])

        # 2) Acceptance probabilities for own infobit only
        lam = params.acceptance_latitude
        k   = params.acceptance_sharpness
        norm = params.max_pxcor + 0.5

        # distance between each guy and *their* new infobit
        D = np.linalg.norm(pos_g - new_pos, axis=1) / norm  # (N,)

        if k == 1:
            p = np.where(D <= 0.0, 1.0, lam / (D + lam))
        else:
            lamk = lam ** k
            Dk = np.where(D <= 0.0, 0.0, D ** k)
            p = np.where(D <= 0.0, 1.0, lamk / (Dk + lamk))

        # whether each guy proposes adding (r -> M0+r)
        accept = self.rng.random(N) < p
        add_rows = np.nonzero(accept)[0]
        if add_rows.size == 0:
            # finalize matrix/columns view and positions, then return
            self.bipartite.H = H
            self.bipartite.invalidate_columns_view()
            self.infobits.old_positions = self.infobits.positions.copy()
            self.infobits.positions = pos_i
            return

        add_cols = M0 + add_rows

        # 3) Enforce memory with "drop-one-uniformly-if-overflow-then-add"
        deg0 = np.asarray(H.sum(axis=1)).ravel().astype(int)
        rem_rows, rem_cols = [], []

        for r, c in zip(add_rows, add_cols):
            will_overflow = (deg0[r] + 1) > params.memory
            if will_overflow and deg0[r] > 0:
                i0, i1 = H.indptr[r], H.indptr[r + 1]
                existing = H.indices[i0:i1]
                # drop exactly one uniformly
                to_drop = self.rng.choice(existing, size=1, replace=False)[0]
                rem_rows.append(r)
                rem_cols.append(to_drop)
                deg0[r] -= 1
            # we don't update deg0[r] for the new edge here; it is added in bulk below

        # apply removals (if any)
        if rem_rows:
            R = sp.csr_array(
                (np.ones(len(rem_rows), dtype=np.uint8),
                (np.array(rem_rows), np.array(rem_cols))),
                shape=H.shape
            )
            H = H - H.multiply(R)

        # 4) Apply additions (all accepted (r, M0+r) pairs)
        A = sp.csr_array(
            (np.ones(add_rows.size, dtype=np.uint8),
            (add_rows, add_cols)),
            shape=(N, pos_i.shape[0])
        )
        if A.shape[1] != H.shape[1]:
            # grow H's columns if needed
            H = sp.hstack([H, sp.csr_array((N, A.shape[1] - H.shape[1]))]).tocsr()
        H = ((H + A) > 0).astype(np.uint8).tocsr()

        # 5) finalize, recompute guy positions as neighbor means
        self.bipartite.H = H
        self.bipartite.invalidate_columns_view()

        deg = np.asarray(H.sum(axis=1)).ravel()
        nz = deg > 0
        if nz.any():
            sums = H @ pos_i
            pos_g[nz] = sums[nz] / deg[nz, None]

        self.infobits.old_positions = self.infobits.positions.copy()
        self.infobits.positions = pos_i


    def try_integrate_infobit(self, guy: Guy, infobit: Infobit, params: Params):
        """ If the infobit is integrated, this updates the guy's position to the mean of 
        the infobit's neighbors' positions. If the infobit is not integrated or it is already
        integrated, it does nothing.
        """
        guy_position = guy.position
        # For now, distance will be a float, but may change as we vectorize more
        distance = float(pairwise_distances(guy_position, infobit.position)) / (params.max_pxcor + 0.5)
        integration_prob = integration_probability(distance, params.acceptance_latitude, params.acceptance_sharpness)
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
        """Replace a friendship with a new friendship with a random friend of a friend.
        """
        for (guy1_id, guy2_id) in self.G.edges():
            guy1 = self.guys[GuyId(guy1_id)]
            guy2 = self.guys[GuyId(guy2_id)]
            distance = float(pairwise_distances(guy1.position, guy2.position)) / (params.max_pxcor + 0.5)
            dislike = 1.0 - integration_probability(distance, params.acceptance_latitude, params.acceptance_sharpness)
            if self.rng.random() < params.refriend_probability * dislike:
                # Choose one of the two guys
                me = GuyId(int(self.rng.choice([guy1_id, guy2_id])))
                # Choose a friend of friend to be friends with
                fof = set(itertools.chain.from_iterable(self.G.neighbors(n) for n in self.G.neighbors(me)))
                fof.discard(me)
                # Only choose fofs that are not already friends
                candidate_friends = [x for x in fof if not self.G.has_edge(me, x)]
                if not candidate_friends:
                    # If no candidate friends in the fof, choose a random guy to befriend
                    new_candidates = [g for g in self.guys if g != me and not self.G.has_edge(me, g)]
                    if not new_candidates:
                        new_friend = None
                    else:
                        new_friend = int(self.rng.choice(new_candidates))
                else:
                    new_friend = int(self.rng.choice(candidate_friends))
                if new_friend is not None:
                    self.G.add_edge(me, new_friend)
                    self.G.remove_edge(guy1_id, guy2_id)

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
    simulation = Simulation.from_params(params)
    simulation.run()
    profiler.disable()
    profiler.dump_stats("naive_profile.prof")



if __name__ == "__main__":
    main()
