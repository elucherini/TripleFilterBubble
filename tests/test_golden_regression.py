"""Regression tests using golden reference data."""
import pytest
import numpy as np
import json
from pathlib import Path
from main import Simulation
from global_params import Params


@pytest.fixture
def golden_data():
    """Load golden reference data."""
    golden_path = Path(__file__).parent / "fixtures" / "golden_seed42_t10.json"

    if not golden_path.exists():
        pytest.skip(f"Golden data not found at {golden_path}. Run generate_golden_data.py first.")

    with open(golden_path, 'r') as f:
        return json.load(f)


@pytest.mark.e2e
class TestGoldenRegression:
    """Tests that compare against golden reference data."""

    def test_exact_match_with_golden_data(self, golden_data, temp_data_dir):
        """Test that simulation produces exact same results as golden data."""
        params = Params(
            seed=golden_data['seed'],
            numguys=golden_data['numguys'],
            numfriends=5,
            numticks=golden_data['numticks'],
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
        assert len(sim.infobits) == golden_data['num_infobits_final'], \
            "Number of infobits differs from golden data"

        assert sim.G.number_of_edges() == golden_data['num_edges_final'], \
            "Number of edges differs from golden data"

        total_info_links = sum(len(sim.H.g2i.get(gid, [])) for gid in sim.guys.keys())
        assert total_info_links == golden_data['total_info_links_final'], \
            "Total info links differs from golden data"

        # Compare final positions (allowing tiny floating point error)
        golden_positions = golden_data['final_positions']
        for gid, guy in sim.guys.items():
            golden_pos = np.array(golden_positions[str(int(gid))])
            np.testing.assert_allclose(
                guy.position, golden_pos,
                rtol=1e-10, atol=1e-10,
                err_msg=f"Position differs from golden data for guy {gid}"
            )

        # Compare edge list
        current_edges = set((int(u), int(v)) if u < v else (int(v), int(u)) for u, v in sim.G.edges())
        golden_edges = set((u, v) if u < v else (v, u) for u, v in golden_data['final_edges'])
        assert current_edges == golden_edges, "Edge list differs from golden data"

    def test_golden_data_metrics_reasonable(self, golden_data):
        """Test that golden data itself has reasonable values."""
        # Basic sanity checks on the golden data
        assert golden_data['num_infobits_final'] > 0
        assert golden_data['num_edges_final'] > 0
        assert golden_data['total_info_links_final'] > 0
        assert golden_data['mean_fluctuation_final'] >= 0.0
        assert golden_data['mean_inf_count_final'] > 0
        assert len(golden_data['final_positions']) == golden_data['numguys']
