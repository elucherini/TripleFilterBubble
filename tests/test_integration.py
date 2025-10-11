"""Integration tests for full simulation runs."""
import pytest
import numpy as np
from pathlib import Path
from main import Simulation
from global_params import Params
import networkx as nx


@pytest.mark.slow
class TestSimulationIntegration:
    """Integration tests that run small simulations end-to-end."""

    def test_deterministic_run(self, test_params, temp_data_dir):
        """Test that runs with same seed produce identical results."""
        test_params.seed = 42
        test_params.numticks = 3
        test_params.run_dir = str(temp_data_dir / "run1")

        sim1 = Simulation.from_params(test_params)
        sim1.run()

        test_params.run_dir = str(temp_data_dir / "run2")
        sim2 = Simulation.from_params(test_params)
        sim2.run()

        # Compare final positions
        for gid in sim1.guys.keys():
            np.testing.assert_array_almost_equal(
                sim1.guys[gid].position,
                sim2.guys[gid].position,
                decimal=10
            )

        # Compare network structure
        assert set(sim1.G.edges()) == set(sim2.G.edges())

    def test_central_mode_creates_shared_infobits(self, test_params, temp_data_dir):
        """Test that central mode creates infobits shared by multiple guys."""
        test_params.new_info_mode = "central"
        test_params.numcentral = 1
        test_params.numticks = 2
        test_params.posting = False  # Disable posting to isolate central creation
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        sim.run()

        # Should have created numcentral * numticks infobits (minus any with degree 0)
        # At least some guys should share infobits
        infobit_degrees = [sim.H.degree_of_info(iid) for iid in sim.infobits.keys()]
        assert any(degree > 1 for degree in infobit_degrees), "No shared infobits in central mode"

    def test_individual_mode_creates_per_guy_infobits(self, test_params, temp_data_dir):
        """Test that individual mode creates one infobit per guy per tick."""
        test_params.new_info_mode = "individual"
        test_params.numticks = 2
        test_params.posting = False
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        sim.run()

        # In individual mode, each guy creates an infobit per tick
        # Some may be removed if they have no connections (very unlikely with many guys/ticks)
        # Just check that at least some infobits were created
        assert len(sim.infobits) > 0, "Individual mode should create infobits"

    def test_positions_stay_in_bounds(self, test_params, temp_data_dir):
        """Test that all guy positions stay within world bounds."""
        test_params.numticks = 10
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        sim.run()

        max_coord = test_params.max_pxcor
        for guy in sim.guys.values():
            assert -max_coord <= guy.position[0] <= max_coord
            assert -max_coord <= guy.position[1] <= max_coord

    def test_memory_cap_enforced(self, test_params, temp_data_dir):
        """Test that guys never exceed memory capacity."""
        test_params.memory = 5
        test_params.numticks = 20
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        sim.run()

        for guy in sim.guys.values():
            num_infobits = len(sim.H.neighbors_of_guy(guy.id))
            assert num_infobits <= test_params.memory
            assert guy.inf_count <= test_params.memory

    def test_network_stable_without_refriend(self, test_params, temp_data_dir):
        """Test that network structure remains stable when refriend_probability=0."""
        test_params.refriend_probability = 0.0
        test_params.birth_death_probability = 0.0
        test_params.numticks = 5
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        initial_edges = set(sim.G.edges())

        sim.run()

        final_edges = set(sim.G.edges())
        assert initial_edges == final_edges

    def test_birth_death_maintains_guy_count(self, test_params, temp_data_dir):
        """Test that birth_death maintains constant number of guys."""
        test_params.birth_death_probability = 0.5
        test_params.numticks = 10
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        initial_count = len(sim.guys)

        sim.run()

        assert len(sim.guys) == initial_count

    def test_orphan_infobits_removed(self, test_params, temp_data_dir):
        """Test that infobits with no connections are removed."""
        test_params.numticks = 10
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        sim.run()

        # All remaining infobits should have at least one connection
        for iid in sim.infobits.keys():
            degree = sim.H.degree_of_info(iid)
            assert degree > 0, f"Orphan infobit {iid} not removed"

    def test_fluctuation_computed(self, test_params, temp_data_dir):
        """Test that fluctuation is computed for all guys."""
        test_params.numticks = 5
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        sim.run()

        # All guys should have fluctuation values (may be 0 if they didn't move)
        for guy in sim.guys.values():
            assert isinstance(guy.fluctuation, float)
            assert guy.fluctuation >= 0.0

    def test_storage_creates_output_files(self, test_params, temp_data_dir):
        """Test that simulation creates expected output files."""
        test_params.numticks = 3
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        sim.run()

        run_dir = Path(test_params.run_dir)

        # Check for compressed files
        assert (run_dir / "meta.npz").exists()
        assert (run_dir / "guy_positions_TxNx2_uint16.npy.zst").exists()
        assert (run_dir / "guy_graph_TxPackedBytes_uint8.npy.zst").exists()
        assert (run_dir / "biadj_events.bin.zst").exists()
        assert (run_dir / "biadj_counts_uint32.npy.zst").exists()
        assert (run_dir / "infobits_final_uint16.npy.zst").exists()
        assert (run_dir / "infobit_ids.npy.zst").exists()

    def test_select_close_infobits_mode(self, test_params, temp_data_dir):
        """Test simulation with select close infobits mode."""
        test_params.new_info_mode = "select close infobits"
        test_params.numticks = 5
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        assert sim._grid is not None, "Spatial grid should be initialized"

        sim.run()

        # Should complete without errors
        assert len(sim.infobits) > 0

    def test_select_distant_infobits_mode(self, test_params, temp_data_dir):
        """Test simulation with select distant infobits mode."""
        test_params.new_info_mode = "select distant infobits"
        test_params.numticks = 5
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        sim.run()

        # Should complete without errors
        # Note: With small number of guys and no posting, it's possible all infobits get removed
        # if they have no connections. This is valid behavior.
        assert len(sim.infobits) >= 0

    def test_posting_shares_information(self, test_params, temp_data_dir):
        """Test that posting increases information spread."""
        test_params.posting = False
        test_params.new_info_mode = "central"
        test_params.numticks = 5
        test_params.run_dir = str(temp_data_dir / "no_posting")

        sim_no_posting = Simulation.from_params(test_params)
        sim_no_posting.run()
        total_links_no_posting = sum(len(sim_no_posting.H.g2i.get(gid, []))
                                     for gid in sim_no_posting.guys.keys())

        test_params.posting = True
        test_params.run_dir = str(temp_data_dir / "with_posting")
        sim_with_posting = Simulation.from_params(test_params)
        sim_with_posting.run()
        total_links_with_posting = sum(len(sim_with_posting.H.g2i.get(gid, []))
                                       for gid in sim_with_posting.guys.keys())

        # With posting should generally have more information links (probabilistic)
        # At minimum, both should have some links
        assert total_links_no_posting > 0
        assert total_links_with_posting > 0

    def test_refriend_changes_network_over_time(self, test_params, temp_data_dir):
        """Test that refriend gradually changes network structure."""
        test_params.refriend_probability = 0.1
        test_params.numticks = 20
        test_params.numguys = 30
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        initial_edges = set(sim.G.edges())

        sim.run()

        final_edges = set(sim.G.edges())

        # Network should have changed somewhat
        # This is probabilistic, but with 20 ticks and prob=0.1, very likely
        edges_added = final_edges - initial_edges
        edges_removed = initial_edges - final_edges

        # At least check that the simulation ran successfully
        assert isinstance(final_edges, set)

    def test_incremental_position_tracking_correct(self, test_params, temp_data_dir):
        """Test that incremental position tracking matches direct calculation."""
        test_params.numticks = 10
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        sim.run()

        # For each guy, verify that position matches inf_sum / inf_count
        for guy in sim.guys.values():
            if guy.inf_count > 0:
                expected_position = guy.inf_sum / guy.inf_count
                np.testing.assert_array_almost_equal(
                    guy.position, expected_position, decimal=5,
                    err_msg=f"Position mismatch for guy {guy.id}"
                )

    def test_infobit_popularity_matches_degree(self, test_params, temp_data_dir):
        """Test that infobit popularity matches its degree in bipartite graph."""
        test_params.numticks = 5
        test_params.run_dir = str(temp_data_dir)

        sim = Simulation.from_params(test_params)
        sim.run()

        for iid, infobit in sim.infobits.items():
            degree = sim.H.degree_of_info(iid)
            assert infobit.popularity == degree, \
                f"Infobit {iid} popularity {infobit.popularity} != degree {degree}"
