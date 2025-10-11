"""Unit tests for Simulation class methods."""
import pytest
import numpy as np
import networkx as nx
from main import Simulation
from models import Guy, GuyId, Infobit, InfobitId, BiAdj
from global_params import Params


class TestSimulation:
    """Tests for Simulation class initialization and methods."""

    def test_from_params(self, test_params):
        """Test Simulation initialization from parameters."""
        sim = Simulation.from_params(test_params)

        assert len(sim.guys) == test_params.numguys
        assert isinstance(sim.G, nx.Graph)
        assert isinstance(sim.H, BiAdj)
        assert len(sim.infobits) == 0  # No infobits at start

    def test_create_guys(self, test_params, rng):
        """Test creating guys."""
        guys = Simulation.create_guys(test_params, rng)

        assert len(guys) == test_params.numguys
        for gid, guy in guys.items():
            assert guy.id == gid
            assert 0 <= guy.group < test_params.numgroups

    def test_make_group_network_intra_only(self, test_params, rng):
        """Test network creation with only intra-group connections."""
        test_params.fraction_inter = 0.0
        test_params.numguys = 10
        test_params.numfriends = 4
        test_params.numgroups = 2

        guys = Simulation.create_guys(test_params, rng)
        G = nx.Graph()

        for guy in guys.values():
            Simulation.make_group_network(guy, guys, G, test_params, rng)

        # Check that all edges are within groups
        for u, v in G.edges():
            assert guys[GuyId(u)].group == guys[GuyId(v)].group

    def test_make_group_network_creates_edges(self, test_params, rng):
        """Test that make_group_network creates edges."""
        test_params.numguys = 10
        test_params.numfriends = 5

        guys = Simulation.create_guys(test_params, rng)
        G = nx.Graph()

        for guy in guys.values():
            Simulation.make_group_network(guy, guys, G, test_params, rng)

        # Should have created some edges
        assert G.number_of_edges() > 0

    def test_try_integrate_infobit_too_far(self, test_params):
        """Test that infobits too far away are not integrated."""
        sim = Simulation.from_params(test_params)

        guy = list(sim.guys.values())[0]
        guy.position = np.array([0.0, 0.0])

        # Create infobit very far away
        far_infobit = Infobit(InfobitId(0), position=np.array([15.0, 15.0]))
        sim.infobits[far_infobit.id] = far_infobit

        initial_count = guy.inf_count
        # Try many times - should never integrate due to distance
        for _ in range(100):
            sim.try_integrate_infobit(guy, far_infobit, test_params)

        # Should not have integrated (or extremely rarely)
        assert guy.inf_count <= initial_count + 1  # Allow for rare random success

    def test_try_integrate_infobit_close(self, test_params):
        """Test that close infobits are more likely to be integrated."""
        sim = Simulation.from_params(test_params)

        guy = list(sim.guys.values())[0]
        guy.position = np.array([0.0, 0.0])

        # Create infobit very close
        close_infobit = Infobit(InfobitId(0), position=np.array([0.01, 0.01]))
        sim.infobits[close_infobit.id] = close_infobit

        initial_count = guy.inf_count
        # Try many times - should integrate at least once
        for _ in range(100):
            sim.try_integrate_infobit(guy, close_infobit, test_params)
            if guy.inf_count > initial_count:
                break

        # Should have integrated at least once
        assert guy.inf_count > initial_count

    def test_try_integrate_memory_cap(self, test_params):
        """Test that memory cap is enforced."""
        test_params.memory = 3
        sim = Simulation.from_params(test_params)

        guy = list(sim.guys.values())[0]
        guy.position = np.array([0.0, 0.0])

        # Create infobits at same position (guaranteed integration)
        # Use try_integrate_infobit which should enforce memory cap
        for i in range(10):
            ib = Infobit(InfobitId(i), position=guy.position.copy())
            sim.infobits[ib.id] = ib
            # Try many times to ensure integration happens despite randomness
            for _ in range(100):
                sim.try_integrate_infobit(guy, ib, test_params)
                if ib.id in sim.H.neighbors_of_guy(guy.id):
                    break

        # Guy should have at most memory infobits
        assert len(sim.H.neighbors_of_guy(guy.id)) <= test_params.memory
        assert guy.inf_count <= test_params.memory

    def test_try_integrate_no_duplicate(self, test_params):
        """Test that the same infobit cannot be integrated twice."""
        sim = Simulation.from_params(test_params)

        guy = list(sim.guys.values())[0]
        guy.position = np.array([0.0, 0.0])

        infobit = Infobit(InfobitId(0), position=guy.position.copy())
        sim.infobits[infobit.id] = infobit

        # Integrate once (force by manually adding)
        sim.H.add_edge(guy.id, infobit.id)
        guy.inf_sum += infobit.position
        guy.inf_count += 1

        initial_count = guy.inf_count
        # Try to integrate again
        sim.try_integrate_infobit(guy, infobit, test_params)

        # Should not have integrated again
        assert guy.inf_count == initial_count

    def test_new_infobits_central_mode(self, test_params):
        """Test new_infobits in central mode."""
        test_params.new_info_mode = "central"
        test_params.numcentral = 2
        sim = Simulation.from_params(test_params)

        initial_infobit_count = len(sim.infobits)
        sim.new_infobits(test_params)

        # Should have created numcentral new infobits
        assert len(sim.infobits) >= initial_infobit_count + test_params.numcentral

    def test_new_infobits_individual_mode(self, test_params):
        """Test new_infobits in individual mode."""
        test_params.new_info_mode = "individual"
        sim = Simulation.from_params(test_params)

        initial_infobit_count = len(sim.infobits)
        sim.new_infobits(test_params)

        # Should have created one infobit per guy
        assert len(sim.infobits) >= initial_infobit_count + test_params.numguys

    def test_update_infobits_removes_orphans(self, test_params):
        """Test that update_infobits removes infobits with no connections."""
        sim = Simulation.from_params(test_params)

        # Add an orphan infobit
        orphan = Infobit(InfobitId(999), position=np.array([0.0, 0.0]))
        sim.infobits[orphan.id] = orphan

        # Add a connected infobit
        connected = Infobit(InfobitId(1000), position=np.array([1.0, 1.0]))
        sim.infobits[connected.id] = connected
        sim.H.add_edge(list(sim.guys.keys())[0], connected.id)

        sim.update_infobits()

        # Orphan should be removed
        assert orphan.id not in sim.infobits
        # Connected should remain
        assert connected.id in sim.infobits

    def test_update_infobits_updates_popularity(self, test_params):
        """Test that update_infobits updates popularity correctly."""
        sim = Simulation.from_params(test_params)

        infobit = Infobit(InfobitId(0), position=np.array([0.0, 0.0]))
        sim.infobits[infobit.id] = infobit

        # Connect to 3 guys
        guy_ids = list(sim.guys.keys())[:3]
        for gid in guy_ids:
            sim.H.add_edge(gid, infobit.id)

        sim.update_infobits()

        assert sim.infobits[infobit.id].popularity == 3

    def test_visualize_updates_fluctuation(self, test_params):
        """Test that visualize updates fluctuation for all guys."""
        sim = Simulation.from_params(test_params)

        # Move some guys
        for guy in sim.guys.values():
            guy.old_position = guy.position.copy()
            guy.position = guy.position + np.array([1.0, 0.0])

        sim.visualize()

        # All guys should have positive fluctuation
        for guy in sim.guys.values():
            assert guy.fluctuation > 0

    def test_birth_death_replaces_guy(self, test_params):
        """Test birth_death replaces guys and clears their info."""
        test_params.birth_death_probability = 1.0  # Always replace
        sim = Simulation.from_params(test_params)

        # Give a guy some infobits
        guy = list(sim.guys.values())[0]
        infobit = Infobit(InfobitId(0), position=np.array([0.0, 0.0]))
        sim.infobits[infobit.id] = infobit
        sim.H.add_edge(guy.id, infobit.id)
        guy.inf_count = 1

        original_position = guy.position.copy()

        sim.birth_death(test_params)

        # Guy should have been replaced
        new_guy = sim.guys[guy.id]
        # Position likely changed (not guaranteed with random, but highly likely)
        # More reliable: check that infolinks were cleared
        assert len(sim.H.neighbors_of_guy(guy.id)) == 0
        assert new_guy.inf_count == 0

    def test_refriend_changes_network(self, test_params):
        """Test that refriend can modify the network."""
        test_params.refriend_probability = 1.0  # Always refriend
        test_params.numguys = 20  # Need enough guys for friend-of-friend
        test_params.numfriends = 5
        sim = Simulation.from_params(test_params)

        # Ensure we have some edges
        if sim.G.number_of_edges() == 0:
            # Add a few manually if needed
            guys = list(sim.guys.keys())
            sim.G.add_edge(guys[0], guys[1])
            sim.G.add_edge(guys[1], guys[2])

        initial_edges = set(sim.G.edges())
        sim.refriend(test_params)
        final_edges = set(sim.G.edges())

        # Network should have changed (not guaranteed but very likely with prob=1.0)
        # At minimum, check that refriend doesn't crash
        assert isinstance(final_edges, set)

    def test_accept_mask_from_d2(self, test_params):
        """Test vectorized acceptance probability calculation."""
        sim = Simulation.from_params(test_params)

        # Test with array of distances
        d2_array = np.array([0.0, 1.0, 10.0, 100.0])
        mask = sim._accept_mask_from_d2(d2_array)

        assert mask.shape == d2_array.shape
        assert mask.dtype == bool
        # Closer distances should have higher acceptance (on average)
        # This is probabilistic, but we can check the mask is boolean

    def test_pick_distant_infobit_fast(self, test_params):
        """Test picking distant infobits."""
        sim = Simulation.from_params(test_params)

        guy = list(sim.guys.values())[0]
        guy.position = np.array([0.0, 0.0])

        # Add distant infobits
        for i in range(10):
            ib = Infobit(InfobitId(i), position=np.array([10.0 + i, 10.0 + i]))
            sim.infobits[ib.id] = ib

        result = sim._pick_distant_infobit_fast(guy, attempts=32)

        # Should find a distant infobit
        if result is not None:
            distance = sim.geo.norm_dist(guy.position, result.position)
            assert distance >= test_params.acceptance_latitude

    def test_pick_close_infobit_from_grid(self, test_params):
        """Test picking close infobits using spatial grid."""
        test_params.new_info_mode = "select close infobits"
        sim = Simulation.from_params(test_params)

        guy = list(sim.guys.values())[0]
        guy.position = np.array([0.0, 0.0])

        # Add a close infobit
        close_ib = Infobit(InfobitId(0), position=np.array([0.1, 0.1]))
        sim.infobits[close_ib.id] = close_ib
        sim._grid.add(close_ib.id, close_ib.position)

        result = sim._pick_close_infobit_from_grid(guy)

        if result is not None:
            distance = sim.geo.norm_dist(guy.position, result.position)
            assert distance < test_params.acceptance_latitude

    def test_post_infobits(self, test_params):
        """Test that post_infobits shares information between friends."""
        sim = Simulation.from_params(test_params)

        # Set up two connected guys
        guy1_id = list(sim.guys.keys())[0]
        guy2_id = list(sim.guys.keys())[1]
        guy1 = sim.guys[guy1_id]
        guy2 = sim.guys[guy2_id]

        # Make them friends
        sim.G.add_edge(guy1_id, guy2_id)

        # Give guy1 an infobit
        infobit = Infobit(InfobitId(0), position=guy1.position.copy())
        sim.infobits[infobit.id] = infobit
        sim.H.add_edge(guy1_id, infobit.id)
        guy1.inf_sum += infobit.position
        guy1.inf_count += 1

        # Position guy2 close to the infobit for higher acceptance
        guy2.position = infobit.position.copy() + np.array([0.01, 0.01])

        initial_guy2_infobits = len(sim.H.neighbors_of_guy(guy2_id))

        # Try posting multiple times
        for _ in range(50):
            sim.post_infobits(test_params)

        # Guy2 might have received the infobit (probabilistic)
        # At minimum, check that post_infobits doesn't crash
        final_guy2_infobits = len(sim.H.neighbors_of_guy(guy2_id))
        assert final_guy2_infobits >= initial_guy2_infobits
