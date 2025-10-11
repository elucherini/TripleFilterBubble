#!/usr/bin/env python3
"""
Example script demonstrating how to use the measurement module.

This script runs a simulation with measurements enabled at specific ticks
and displays the results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from global_params import Params
from main import Simulation


def main():
    # Configure parameters with measurement ticks
    params = Params(
        numguys=100,
        numfriends=10,
        numgroups=4,
        numticks=50,
        memory=20,
        acceptance_latitude=0.3,
        acceptance_sharpness=20.0,
        max_pxcor=16.0,
        seed=42,
        posting=True,
        new_info_mode="central",
        numcentral=1,
        # Measure at ticks 10, 25, and 49
        measurement_ticks=[10, 25, 49],
        run_dir="data_with_measurements"
    )

    print("=" * 70)
    print("Running TripleFilterBubble Simulation with Measurements")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  - Guys: {params.numguys}")
    print(f"  - Friends per guy: {params.numfriends}")
    print(f"  - Groups: {params.numgroups}")
    print(f"  - Ticks: {params.numticks}")
    print(f"  - Measurement ticks: {params.measurement_ticks}")
    print("=" * 70)
    print()

    # Create and run simulation
    sim = Simulation.from_params(params, enable_plotting=False)
    sim.run()

    # Measurements are automatically printed at the end of run()
    # But you can also access them programmatically:
    if sim.measurements:
        print("\nProgrammatic access to measurements:")
        print("-" * 70)
        for tick in sorted(sim.measurements.mean_link_length.keys()):
            ll = sim.measurements.mean_link_length[tick]
            isd = sim.measurements.mean_infosharer_distance[tick]
            fd = sim.measurements.mean_friend_distance[tick]

            print(f"Tick {tick}:")
            if ll is not None:
                print(f"  Link Length: {ll:.6f}")
            if isd is not None:
                print(f"  Infosharer Distance: {isd:.6f}")
            if fd is not None:
                print(f"  Friend Distance: {fd:.6f}")
            print()


if __name__ == "__main__":
    main()
