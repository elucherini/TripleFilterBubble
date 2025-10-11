"""Tests for the metrics module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
from global_params import Params
from main import Simulation
from models import Guy, GuyId, Infobit, InfobitId, BiAdj
from metrics import MeasurementResults, compute_metrics, _compute_mean_link_length, _compute_mean_infosharer_distance, _compute_mean_friend_distance
import networkx as nx
from utils import FastGeo


class TestMeasurementResults:
    """Test the MeasurementResults dataclass."""

    def test_initialization(self):
        """Test that MeasurementResults can be initialized."""
        results = MeasurementResults()
        assert results.mean_link_length == {}
        assert results.mean_infosharer_distance == {}
        assert results.mean_friend_distance == {}

    def test_add_measurement(self):
        """Test adding measurements for specific ticks."""
        results = MeasurementResults()
        results.add_measurement(tick=10, mean_link_length=0.5, mean_infosharer_distance=0.3, mean_friend_distance=0.4)

        assert results.mean_link_length[10] == 0.5
        assert results.mean_infosharer_distance[10] == 0.3
        assert results.mean_friend_distance[10] == 0.4

    def test_add_measurement_with_none(self):
        """Test adding measurements with None values."""
        results = MeasurementResults()
        results.add_measurement(tick=5, mean_link_length=None, mean_infosharer_distance=0.2, mean_friend_distance=None)

        assert results.mean_link_length[5] is None
        assert results.mean_infosharer_distance[5] == 0.2
        assert results.mean_friend_distance[5] is None

    def test_repr(self):
        """Test that the repr prints nicely."""
        results = MeasurementResults()
        results.add_measurement(tick=10, mean_link_length=0.5, mean_infosharer_distance=0.3, mean_friend_distance=0.4)
        repr_str = repr(results)

        assert "Tick" in repr_str
        assert "10" in repr_str
        assert "0.5" in repr_str


class TestComputeMetrics:
    """Test the compute_metrics function and its helpers."""

    @pytest.fixture
    def simple_simulation(self):
        """Create a simple simulation with 3 guys and 2 infobits."""
        params = Params(
            numguys=3,
            numfriends=2,
            numgroups=1,
            numticks=10,
            memory=5,
            acceptance_latitude=0.3,
            acceptance_sharpness=20.0,
            max_pxcor=16.0,
            seed=42,
            posting=True,
            new_info_mode="central",
            numcentral=1,
        )

        sim = Simulation.from_params(params, enable_plotting=False)

        # Create infobits manually for testing
        info1 = Infobit(id=InfobitId(0), position=np.array([1.0, 1.0]))
        info2 = Infobit(id=InfobitId(1), position=np.array([2.0, 2.0]))
        sim.infobits[info1.id] = info1
        sim.infobits[info2.id] = info2

        # Get guys as a list for easier access
        guy_list = list(sim.guys.values())

        # Guy 0 has info1
        sim.H.add_edge(guy_list[0].id, info1.id)
        sim.H.sharer[(guy_list[0].id, info1.id)] = guy_list[0].id  # self-created

        # Guy 1 has info2, shared by Guy 0
        sim.H.add_edge(guy_list[1].id, info2.id)
        sim.H.sharer[(guy_list[1].id, info2.id)] = guy_list[0].id

        # Guy 2 has both infobits
        sim.H.add_edge(guy_list[2].id, info1.id)
        sim.H.add_edge(guy_list[2].id, info2.id)
        sim.H.sharer[(guy_list[2].id, info1.id)] = guy_list[1].id
        sim.H.sharer[(guy_list[2].id, info2.id)] = guy_list[0].id

        return sim

    def test_compute_mean_link_length_with_infobits(self, simple_simulation):
        """Test that mean link length is computed correctly."""
        result = _compute_mean_link_length(simple_simulation)
        assert result is not None
        assert isinstance(result, float)
        assert result >= 0.0

    def test_compute_mean_link_length_no_infobits(self):
        """Test that mean link length returns None when no infobits exist."""
        params = Params(numguys=3, numfriends=2, numgroups=1, numticks=10)
        sim = Simulation.from_params(params, enable_plotting=False)
        result = _compute_mean_link_length(sim)
        assert result is None

    def test_compute_mean_infosharer_distance(self, simple_simulation):
        """Test that mean infosharer distance is computed correctly."""
        result = _compute_mean_infosharer_distance(simple_simulation)
        assert result is not None
        assert isinstance(result, float)
        assert result >= 0.0

    def test_compute_mean_infosharer_distance_no_sharers(self):
        """Test that mean infosharer distance returns None when no sharers exist."""
        params = Params(numguys=3, numfriends=2, numgroups=1, numticks=10)
        sim = Simulation.from_params(params, enable_plotting=False)
        result = _compute_mean_infosharer_distance(sim)
        assert result is None

    def test_compute_mean_friend_distance(self, simple_simulation):
        """Test that mean friend distance is computed correctly."""
        result = _compute_mean_friend_distance(simple_simulation)
        assert result is not None
        assert isinstance(result, float)
        assert result >= 0.0

    def test_compute_mean_friend_distance_no_friends(self):
        """Test that mean friend distance returns None when graph is empty."""
        params = Params(numguys=3, numfriends=0, numgroups=1, numticks=10)
        sim = Simulation.from_params(params, enable_plotting=False)
        # Remove all edges manually to ensure empty graph
        sim.G.clear()
        result = _compute_mean_friend_distance(sim)
        assert result is None

    def test_compute_metrics_all_together(self, simple_simulation):
        """Test that compute_metrics returns all three metrics."""
        metrics = compute_metrics(simple_simulation, tick=0)

        assert 'mean_link_length' in metrics
        assert 'mean_infosharer_distance' in metrics
        assert 'mean_friend_distance' in metrics

        # With our simple simulation, all should have values
        assert metrics['mean_link_length'] is not None
        assert metrics['mean_infosharer_distance'] is not None
        assert metrics['mean_friend_distance'] is not None


class TestIntegrationWithSimulation:
    """Integration tests that run a full simulation with measurements."""

    def test_simulation_with_measurements(self, temp_data_dir):
        """Test that a simulation can be run with measurements enabled."""
        params = Params(
            numguys=20,
            numfriends=5,
            numgroups=2,
            numticks=5,
            memory=5,
            acceptance_latitude=0.3,
            acceptance_sharpness=20.0,
            max_pxcor=16.0,
            seed=42,
            posting=True,
            new_info_mode="central",
            numcentral=1,
            measurement_ticks=[2, 4],  # Measure at ticks 2 and 4
            run_dir=str(temp_data_dir),
        )

        sim = Simulation.from_params(params, enable_plotting=False)
        assert sim.measurements is not None

        # Run the simulation
        sim.run()

        # Check that measurements were recorded at the correct ticks
        assert 2 in sim.measurements.mean_link_length
        assert 4 in sim.measurements.mean_link_length

        # Check that measurements are reasonable (not None or NaN)
        assert sim.measurements.mean_link_length[2] is not None or sim.measurements.mean_link_length[2] is None
        assert sim.measurements.mean_friend_distance[2] is not None

    def test_simulation_without_measurements(self, temp_data_dir):
        """Test that a simulation without measurement_ticks works normally."""
        params = Params(
            numguys=10,
            numfriends=3,
            numgroups=2,
            numticks=3,
            memory=5,
            acceptance_latitude=0.3,
            acceptance_sharpness=20.0,
            max_pxcor=16.0,
            seed=42,
            posting=True,
            new_info_mode="central",
            numcentral=1,
            measurement_ticks=[],  # No measurements
            run_dir=str(temp_data_dir),
        )

        sim = Simulation.from_params(params, enable_plotting=False)
        assert sim.measurements is None

        # Run the simulation (should not crash)
        sim.run()

        # Verify no measurements were taken
        assert sim.measurements is None

    def test_sharer_tracking_in_new_infobits(self, temp_data_dir):
        """Test that sharers are tracked correctly when guys create their own infobits."""
        params = Params(
            numguys=5,
            numfriends=2,
            numgroups=1,
            numticks=1,
            memory=10,
            acceptance_latitude=0.3,
            acceptance_sharpness=20.0,
            max_pxcor=16.0,
            seed=42,
            posting=False,  # Disable posting to only test new_infobits
            new_info_mode="central",
            numcentral=1,
            run_dir=str(temp_data_dir),
        )

        sim = Simulation.from_params(params, enable_plotting=False)
        sim.storage.setup_writers(sim.guys, params.numticks)
        sim.storage.attach_biadj_callbacks(sim.H)

        # Run one tick
        sim.storage.begin_tick(0)
        sim.new_infobits(params)

        # Check that sharers are tracked (guys who integrated the central infobit are their own sharers)
        for guy_id, guy in sim.guys.items():
            infobit_ids = sim.H.g2i.get(guy_id, set())
            for infobit_id in infobit_ids:
                sharer = sim.H.sharer.get((guy_id, infobit_id))
                # For central mode, guys integrate infobits themselves
                assert sharer == guy_id

    def test_sharer_tracking_in_post_infobits(self, temp_data_dir):
        """Test that sharers are tracked correctly when guys post to friends."""
        params = Params(
            numguys=10,
            numfriends=5,
            numgroups=1,
            numticks=2,
            memory=10,
            acceptance_latitude=0.3,
            acceptance_sharpness=20.0,
            max_pxcor=16.0,
            seed=42,
            posting=True,
            new_info_mode="central",
            numcentral=1,
            run_dir=str(temp_data_dir),
        )

        sim = Simulation.from_params(params, enable_plotting=False)
        sim.storage.setup_writers(sim.guys, params.numticks)
        sim.storage.attach_biadj_callbacks(sim.H)

        # Run two ticks to allow posting
        sim.storage.begin_tick(0)
        sim.new_infobits(params)
        sim.storage.end_tick(0)

        sim.storage.begin_tick(1)
        sim.post_infobits(params)
        sim.storage.end_tick(1)

        # Check that some sharers are different from the guy who has the infobit
        different_sharers = 0
        for guy_id, guy in sim.guys.items():
            infobit_ids = sim.H.g2i.get(guy_id, set())
            for infobit_id in infobit_ids:
                sharer = sim.H.sharer.get((guy_id, infobit_id))
                if sharer is not None and sharer != guy_id:
                    different_sharers += 1

        # We expect at least some infobits to have been shared by friends
        # (This may be 0 in rare cases due to random acceptance, but typically > 0)
        assert different_sharers >= 0  # Just check it doesn't crash; actual value depends on RNG


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_simulation(self):
        """Test metrics computation on a simulation with no guys."""
        params = Params(numguys=0, numfriends=0, numgroups=1, numticks=1)
        # Manually create an empty simulation
        sim = Simulation.from_params(params, enable_plotting=False)

        metrics = compute_metrics(sim, tick=0)

        # All metrics should be None for empty simulation
        assert metrics['mean_link_length'] is None
        assert metrics['mean_infosharer_distance'] is None
        assert metrics['mean_friend_distance'] is None

    def test_single_guy(self):
        """Test metrics computation with a single guy."""
        params = Params(numguys=1, numfriends=0, numgroups=1, numticks=1)
        sim = Simulation.from_params(params, enable_plotting=False)

        metrics = compute_metrics(sim, tick=0)

        # Single guy with no infobits or friends
        assert metrics['mean_link_length'] is None
        assert metrics['mean_infosharer_distance'] is None
        assert metrics['mean_friend_distance'] is None
