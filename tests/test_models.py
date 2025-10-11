"""Unit tests for Guy, Infobit, and BiAdj models."""
import pytest
import numpy as np
from models import Guy, GuyId, Infobit, InfobitId, BiAdj
from global_params import Params
import math


class TestGuy:
    """Tests for Guy agent class."""

    def test_random_setup(self, test_params, rng):
        """Test random Guy initialization."""
        guy = Guy.random_setup(0, test_params, rng)

        assert guy.id == GuyId(0)
        assert 0 <= guy.group < test_params.numgroups
        assert -test_params.max_pxcor <= guy.position[0] <= test_params.max_pxcor
        assert -test_params.max_pxcor <= guy.position[1] <= test_params.max_pxcor
        assert guy.fluctuation == 0.0
        assert guy.inf_count == 0
        np.testing.assert_array_equal(guy.inf_sum, np.zeros(2))

    def test_guy_initialization(self):
        """Test direct Guy initialization."""
        pos = np.array([1.0, 2.0])
        guy = Guy(id=GuyId(5), position=pos, group=1)

        assert guy.id == GuyId(5)
        assert guy.group == 1
        np.testing.assert_array_equal(guy.position, pos)
        np.testing.assert_array_equal(guy.old_position, pos)

    def test_update_fluctuation_no_movement(self):
        """Test fluctuation calculation when Guy doesn't move."""
        guy = Guy(id=GuyId(0), position=np.array([0.0, 0.0]), group=0)
        guy.old_position = guy.position.copy()

        guy.update_fluctuation(max_pxcor=16.0, max_pycor=16.0)

        assert guy.fluctuation == 0.0

    def test_update_fluctuation_simple_movement(self):
        """Test fluctuation calculation for simple movement."""
        guy = Guy(id=GuyId(0), position=np.array([1.0, 0.0]), group=0)
        guy.old_position = np.array([0.0, 0.0])

        guy.update_fluctuation(max_pxcor=16.0, max_pycor=16.0)

        # Distance = 1.0, half_x = 16.5
        # Fluctuation = 1.0 / 16.5
        expected = 1.0 / 16.5
        assert abs(guy.fluctuation - expected) < 1e-10

    def test_update_fluctuation_diagonal_movement(self):
        """Test fluctuation calculation for diagonal movement."""
        guy = Guy(id=GuyId(0), position=np.array([3.0, 4.0]), group=0)
        guy.old_position = np.array([0.0, 0.0])

        guy.update_fluctuation(max_pxcor=16.0, max_pycor=16.0)

        # Distance = sqrt(3^2 + 4^2) = 5.0, half_x = 16.5
        # Fluctuation = 5.0 / 16.5
        expected = 5.0 / 16.5
        assert abs(guy.fluctuation - expected) < 1e-10

    def test_update_fluctuation_torus_wrapping(self):
        """Test fluctuation calculation with torus wrapping."""
        max_pxcor = 16.0
        guy = Guy(id=GuyId(0), position=np.array([15.5, 0.0]), group=0)
        guy.old_position = np.array([-15.5, 0.0])

        guy.update_fluctuation(max_pxcor=max_pxcor, max_pycor=max_pxcor)

        # Straight-line distance would be 31.0
        # But with torus wrapping: min(31.0, 33.0 - 31.0) = min(31.0, 2.0) = 2.0
        # Fluctuation = 2.0 / 16.5
        expected = 2.0 / 16.5
        assert abs(guy.fluctuation - expected) < 1e-9

    def test_incremental_position_tracking(self):
        """Test that inf_sum and inf_count track position updates correctly."""
        guy = Guy(id=GuyId(0), position=np.array([0.0, 0.0]), group=0)

        # Simulate integrating two infobits
        infobit1_pos = np.array([1.0, 0.0])
        infobit2_pos = np.array([0.0, 1.0])

        guy.inf_sum += infobit1_pos
        guy.inf_count += 1
        guy.position[:] = guy.inf_sum / guy.inf_count

        np.testing.assert_array_almost_equal(guy.position, np.array([1.0, 0.0]))

        guy.inf_sum += infobit2_pos
        guy.inf_count += 1
        guy.position[:] = guy.inf_sum / guy.inf_count

        np.testing.assert_array_almost_equal(guy.position, np.array([0.5, 0.5]))

    def test_old_position_updated(self):
        """Test that update_fluctuation updates old_position."""
        guy = Guy(id=GuyId(0), position=np.array([5.0, 5.0]), group=0)
        guy.old_position = np.array([0.0, 0.0])

        guy.update_fluctuation(max_pxcor=16.0, max_pycor=16.0)

        np.testing.assert_array_equal(guy.old_position, np.array([5.0, 5.0]))


class TestInfobit:
    """Tests for Infobit information piece class."""

    def test_random_setup(self, test_params, rng):
        """Test random Infobit initialization."""
        infobit = Infobit.random_setup(0, test_params, rng)

        assert infobit.id == InfobitId(0)
        assert -test_params.max_pxcor <= infobit.position[0] <= test_params.max_pxcor
        assert -test_params.max_pxcor <= infobit.position[1] <= test_params.max_pxcor
        assert infobit.popularity == 0.0

    def test_infobit_initialization(self):
        """Test direct Infobit initialization."""
        pos = np.array([3.0, -2.0])
        infobit = Infobit(id=InfobitId(10), position=pos, popularity=5.0)

        assert infobit.id == InfobitId(10)
        np.testing.assert_array_equal(infobit.position, pos)
        assert infobit.popularity == 5.0
        np.testing.assert_array_equal(infobit.old_position, pos)

    def test_default_popularity(self):
        """Test that default popularity is 0.0."""
        infobit = Infobit(id=InfobitId(0), position=np.array([0.0, 0.0]))
        assert infobit.popularity == 0.0


class TestBiAdj:
    """Tests for BiAdj bidirectional adjacency structure."""

    def test_empty_initialization(self):
        """Test BiAdj starts empty."""
        bi = BiAdj()
        assert len(bi.g2i) == 0
        assert len(bi.i2g) == 0

    def test_add_edge_basic(self, biadj):
        """Test adding a single edge."""
        gid = GuyId(0)
        iid = InfobitId(0)

        result = biadj.add_edge(gid, iid)

        assert result is True
        assert iid in biadj.g2i[gid]
        assert gid in biadj.i2g[iid]

    def test_add_edge_duplicate(self, biadj):
        """Test adding the same edge twice."""
        gid = GuyId(0)
        iid = InfobitId(0)

        result1 = biadj.add_edge(gid, iid)
        result2 = biadj.add_edge(gid, iid)

        assert result1 is True
        assert result2 is False  # Should return False for duplicate
        assert len(biadj.g2i[gid]) == 1

    def test_remove_edge_basic(self, biadj):
        """Test removing an edge."""
        gid = GuyId(0)
        iid = InfobitId(0)

        biadj.add_edge(gid, iid)
        result = biadj.remove_edge(gid, iid)

        assert result is True
        assert gid not in biadj.g2i or iid not in biadj.g2i[gid]
        assert iid not in biadj.i2g or gid not in biadj.i2g[iid]

    def test_remove_nonexistent_edge(self, biadj):
        """Test removing an edge that doesn't exist."""
        gid = GuyId(0)
        iid = InfobitId(0)

        result = biadj.remove_edge(gid, iid)

        assert result is False

    def test_neighbors_of_guy(self, biadj):
        """Test getting all infobits connected to a guy."""
        gid = GuyId(0)
        iid1 = InfobitId(1)
        iid2 = InfobitId(2)
        iid3 = InfobitId(3)

        biadj.add_edge(gid, iid1)
        biadj.add_edge(gid, iid2)
        biadj.add_edge(gid, iid3)

        neighbors = biadj.neighbors_of_guy(gid)

        assert len(neighbors) == 3
        assert iid1 in neighbors
        assert iid2 in neighbors
        assert iid3 in neighbors

    def test_neighbors_of_nonexistent_guy(self, biadj):
        """Test getting neighbors of a guy with no edges."""
        neighbors = biadj.neighbors_of_guy(GuyId(999))
        assert len(neighbors) == 0

    def test_degree_of_info(self, biadj):
        """Test getting degree of an infobit."""
        iid = InfobitId(0)
        gid1 = GuyId(1)
        gid2 = GuyId(2)
        gid3 = GuyId(3)

        biadj.add_edge(gid1, iid)
        biadj.add_edge(gid2, iid)
        biadj.add_edge(gid3, iid)

        degree = biadj.degree_of_info(iid)
        assert degree == 3

    def test_degree_of_nonexistent_info(self, biadj):
        """Test degree of infobit with no edges."""
        degree = biadj.degree_of_info(InfobitId(999))
        assert degree == 0

    def test_callbacks_on_add(self, biadj):
        """Test that callbacks are called when adding edges."""
        added_edges = []

        def on_add(gid, iid):
            added_edges.append((gid, iid))

        biadj._on_add = on_add

        gid = GuyId(0)
        iid = InfobitId(0)
        biadj.add_edge(gid, iid)

        assert len(added_edges) == 1
        assert added_edges[0] == (gid, iid)

    def test_callbacks_on_remove(self, biadj):
        """Test that callbacks are called when removing edges."""
        removed_edges = []

        def on_remove(gid, iid):
            removed_edges.append((gid, iid))

        biadj._on_remove = on_remove

        gid = GuyId(0)
        iid = InfobitId(0)
        biadj.add_edge(gid, iid)
        biadj.remove_edge(gid, iid)

        assert len(removed_edges) == 1
        assert removed_edges[0] == (gid, iid)

    def test_drop_all_for_guy(self, biadj):
        """Test dropping all edges for a guy."""
        gid = GuyId(0)
        iid1 = InfobitId(1)
        iid2 = InfobitId(2)

        biadj.add_edge(gid, iid1)
        biadj.add_edge(gid, iid2)

        biadj.drop_all_for_guy(gid)

        assert len(biadj.neighbors_of_guy(gid)) == 0
        # Infobit sides should also be cleaned up
        assert gid not in biadj.i2g.get(iid1, set())
        assert gid not in biadj.i2g.get(iid2, set())

    def test_multiple_guys_one_infobit(self, biadj):
        """Test multiple guys connected to one infobit."""
        iid = InfobitId(0)
        guys = [GuyId(i) for i in range(5)]

        for gid in guys:
            biadj.add_edge(gid, iid)

        assert biadj.degree_of_info(iid) == 5

        for gid in guys:
            neighbors = biadj.neighbors_of_guy(gid)
            assert iid in neighbors

    def test_cleanup_on_remove(self, biadj):
        """Test that empty sets are removed from dictionaries."""
        gid = GuyId(0)
        iid = InfobitId(0)

        biadj.add_edge(gid, iid)
        biadj.remove_edge(gid, iid)

        # Empty entries should be removed
        assert gid not in biadj.g2i
        assert iid not in biadj.i2g
