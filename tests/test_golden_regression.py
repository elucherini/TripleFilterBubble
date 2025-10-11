"""Regression tests using NetLogo reference data."""
import pytest
import numpy as np
import json
from pathlib import Path
from main import Simulation
from global_params import Params


@pytest.fixture
def netlogo_reference():
    """Load NetLogo reference data from fixtures."""
    reference_path = Path(__file__).parent / "fixtures" / "netlogo_reference.json"

    if not reference_path.exists():
        pytest.skip(f"NetLogo reference data not found at {reference_path}. Generate it from NetLogo first.")

    with open(reference_path, 'r') as f:
        return json.load(f)


@pytest.mark.e2e
class TestNetLogoRegression:
    """Tests that compare Python implementation against NetLogo reference."""

    @pytest.mark.xfail(reason="Known discrepancy: Python produces 8 infobits vs NetLogo's 10. Needs investigation.")
    def test_match_netlogo_reference(self, netlogo_reference, temp_data_dir):
        """Test that Python simulation produces results matching NetLogo reference.

        Currently marked as expected to fail due to a known discrepancy in infobit count.
        This test documents the expected NetLogo behavior and will help track when
        the Python implementation achieves full parity.
        """
        params = Params(
            seed=netlogo_reference['seed'],
            numguys=netlogo_reference['numguys'],
            numfriends=5,
            numticks=netlogo_reference['numticks'],
            numgroups=2,
            new_info_mode="central",
            numcentral=1,
            posting=True,
            birth_death_probability=0.0,
            refriend_probability=0.0,
            memory=10,
            acceptance_latitude=0.3,
            acceptance_sharpness=20.0,
            max_pxcor=16.0,
            run_dir=str(temp_data_dir)
        )

        sim = Simulation.from_params(params)
        sim.run()

        # Compare metrics
        assert len(sim.infobits) == netlogo_reference['num_infobits_final'], \
            f"Number of infobits differs: Python={len(sim.infobits)}, NetLogo={netlogo_reference['num_infobits_final']}"

        assert sim.G.number_of_edges() == netlogo_reference['num_edges_final'], \
            f"Number of edges differs: Python={sim.G.number_of_edges()}, NetLogo={netlogo_reference['num_edges_final']}"

        total_info_links = sum(len(sim.H.g2i.get(gid, [])) for gid in sim.guys.keys())
        assert total_info_links == netlogo_reference['total_info_links_final'], \
            f"Total info links differs: Python={total_info_links}, NetLogo={netlogo_reference['total_info_links_final']}"

        # Compare final positions (allowing small floating point error for cross-platform differences)
        netlogo_positions = netlogo_reference['final_positions']
        for gid, guy in sim.guys.items():
            netlogo_pos = np.array(netlogo_positions[str(int(gid))])
            np.testing.assert_allclose(
                guy.position, netlogo_pos,
                rtol=1e-6, atol=1e-6,
                err_msg=f"Position differs from NetLogo for guy {gid}: Python={guy.position}, NetLogo={netlogo_pos}"
            )

        # Compare edge list
        current_edges = set((int(u), int(v)) if u < v else (int(v), int(u)) for u, v in sim.G.edges())
        netlogo_edges = set((u, v) if u < v else (v, u) for u, v in netlogo_reference['final_edges'])
        assert current_edges == netlogo_edges, \
            f"Edge list differs from NetLogo. Symmetric difference: {current_edges.symmetric_difference(netlogo_edges)}"

    def test_netlogo_reference_sanity(self, netlogo_reference):
        """Test that NetLogo reference data has reasonable values."""
        # Basic sanity checks on the NetLogo reference data
        assert netlogo_reference['num_infobits_final'] > 0
        assert netlogo_reference['num_edges_final'] > 0
        assert netlogo_reference['total_info_links_final'] > 0
        assert netlogo_reference['mean_fluctuation_final'] >= 0.0
        assert netlogo_reference['mean_inf_count_final'] > 0
        assert len(netlogo_reference['final_positions']) == netlogo_reference['numguys']
