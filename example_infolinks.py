"""
Example script demonstrating infolink visualization.

This script creates comparison plots showing:
1. Baseline: agents only (no infobits or infolinks)
2. With infobits: agents and infobits
3. With infolinks: agents, infobits, and connections between them

This allows visual verification that infolinks are properly displayed.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import Simulation
from global_params import Params

def main():
    # Create simulation with plotting enabled
    params = Params()
    params.numticks = 5  # Just a few ticks to create some interesting connections
    params.numguys = 50  # Fewer agents for clearer visualization
    params.memory = 10   # Smaller memory to keep infolinks manageable
    params.show_infobits = True
    params.show_infolinks = True
    params.infobit_size = True

    print("Creating simulation with plotting enabled...")
    model = Simulation.from_params(params, enable_plotting=True)

    # Run simulation to create some connections
    print(f"Running simulation for {params.numticks} ticks...")
    model.storage.setup_writers(model.guys, params.numticks)
    if model.params.refriend_probability == 0:
        model.storage.precompute_guy_graph_row(model.G)
    model.storage.attach_biadj_callbacks(model.H)

    for tick in range(params.numticks):
        model.storage.begin_tick(tick)
        model.new_infobits(model.params)
        if model.params.posting:
            model.post_infobits(model.params)
        if model.params.birth_death_probability > 0:
            model.birth_death(model.params)
        if model.params.refriend_probability > 0:
            model.refriend(model.params)
        model.update_infobits()
        model.visualize()
        model.storage.write_tick(tick, model.guys)
        model.storage.write_guy_graph(tick, model.G)
        model.storage.end_tick(tick)

    model.storage.finalize(model.infobits)

    print(f"\nSimulation complete. Creating comparison plots...")
    print(f"Total infobits: {len(model.infobits)}")
    print(f"Total infolinks: {sum(len(v) for v in model.H.g2i.values())}")

    # 1. Baseline: agents only
    print("\n1. Creating baseline plot (agents only)...")
    model.plot_current_positions(
        title="Baseline: Agents Only",
        color_by_group=True,
        save_path="comparison_1_baseline.png",
        show_infolinks=False
    )

    # 2. With infobits only
    print("2. Creating plot with infobits...")
    model.plot_current_positions(
        title="Agents with Infobits (No Links)",
        color_by_group=True,
        save_path="comparison_2_with_infobits.png",
        show_infolinks=False
    )

    # 3. With infobits and infolinks
    print("3. Creating plot with infobits and infolinks...")
    model.plot_current_positions(
        title="Agents with Infobits and Infolinks",
        color_by_group=True,
        save_path="comparison_3_with_infolinks.png",
        show_infolinks=True
    )

    print("\n" + "="*60)
    print("Comparison plots saved successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - comparison_1_baseline.png (agents only)")
    print("  - comparison_2_with_infobits.png (agents + infobits)")
    print("  - comparison_3_with_infolinks.png (agents + infobits + connections)")
    print("\nVisualization summary:")
    print(f"  - {params.numguys} agents across {params.numgroups} groups")
    print(f"  - {len(model.infobits)} infobits created")
    print(f"  - {sum(len(v) for v in model.H.g2i.values())} infolinks (guy→infobit connections)")
    print(f"  - Infolinks drawn as thin gray lines with alpha=0.15")
    print("\nPlease visually verify that:")
    print("  1. Baseline shows only colored agent dots")
    print("  2. With infobits shows gray squares added")
    print("  3. With infolinks shows thin gray lines connecting agents to infobits")

if __name__ == "__main__":
    main()
