#!/usr/bin/env python3
"""
Example script to demonstrate friend link visualization.

This script runs a small simulation with friend links, infolinks, and infobits
all visible to verify the visualization functionality.
"""

import sys
sys.path.insert(0, 'src')

from global_params import Params
from main import Simulation


def main():
    # Create params with visualization enabled
    params = Params(
        # Small simulation for quick testing
        numguys=50,
        numticks=10,
        numfriends=5,
        seed=42,

        # Enable all visualizations
        show_infobits=True,
        show_infolinks=True,
        show_friend_links=True,
        infobit_size=True,

        # Plot at start, middle, and end
        plot_every_n_ticks=5,

        # Use central info mode for clearer visualization
        new_info_mode="central",
        numcentral=1,

        # Disable birth/death and refriending for stable network
        birth_death_probability=0.0,
        refriend_probability=0.0,

        # Moderate acceptance for visible clustering
        acceptance_latitude=0.3,

        # Output directory
        run_dir="data/example_friend_links"
    )

    print("Running simulation with friend link visualization...")
    print(f"Parameters:")
    print(f"  - {params.numguys} agents")
    print(f"  - {params.numticks} ticks")
    print(f"  - {params.numfriends} friends per agent")
    print(f"  - Plotting every {params.plot_every_n_ticks} ticks")
    print()

    # Create and run simulation
    model = Simulation.from_params(params, enable_plotting=True)
    model.run()

    print()
    print("Simulation complete!")
    print("Check the generated PNG files to see:")
    print("  - Blue lines: friend links (linewidth=2)")
    print("  - Gray lines: infolinks (connections to infobits)")
    print("  - Gray squares: infobits (sized by popularity)")
    print("  - Colored dots: agents (colored by group)")


if __name__ == "__main__":
    main()
