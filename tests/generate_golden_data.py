"""
Generate golden reference data for regression testing.

This script runs a small simulation with fixed parameters and saves
key metrics that can be used to detect regressions in future runs.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import simulation modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from main import Simulation
from global_params import Params
import json


def generate_golden_data():
    """Generate golden reference data for testing."""
    print("Generating golden reference data...")

    # Use small, fast parameters for golden data
    params = Params(
        seed=42,
        numguys=20,
        numfriends=5,
        numticks=10,
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
        run_dir="tests/fixtures/golden_seed42_t10"
    )

    # Create simulation and run
    sim = Simulation.from_params(params)
    sim.run()

    # Collect metrics
    golden_data = {
        'seed': params.seed,
        'numguys': params.numguys,
        'numticks': params.numticks,
        'num_infobits_final': len(sim.infobits),
        'num_edges_final': sim.G.number_of_edges(),
        'total_info_links_final': sum(len(sim.H.g2i.get(gid, [])) for gid in sim.guys.keys()),
        'mean_fluctuation_final': float(np.mean([g.fluctuation for g in sim.guys.values()])),
        'mean_inf_count_final': float(np.mean([g.inf_count for g in sim.guys.values()])),
        'mean_infobit_popularity': float(np.mean([ib.popularity for ib in sim.infobits.values()])) if sim.infobits else 0.0,
    }

    # Save final positions for exact comparison
    final_positions = {}
    for gid, guy in sim.guys.items():
        final_positions[int(gid)] = guy.position.tolist()

    golden_data['final_positions'] = final_positions

    # Save edge list
    golden_data['final_edges'] = [[int(u), int(v)] for u, v in sim.G.edges()]

    # Save to JSON
    output_path = Path("tests/fixtures/golden_seed42_t10.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(golden_data, f, indent=2)

    print(f"Golden data saved to {output_path}")
    print(f"Summary:")
    print(f"  - Infobits: {golden_data['num_infobits_final']}")
    print(f"  - Edges: {golden_data['num_edges_final']}")
    print(f"  - Info links: {golden_data['total_info_links_final']}")
    print(f"  - Mean fluctuation: {golden_data['mean_fluctuation_final']:.6f}")


if __name__ == "__main__":
    generate_golden_data()
