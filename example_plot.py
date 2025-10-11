"""
Example script demonstrating the PositionPlotter functionality.

This shows how to:
1. Create a simulation with plotting enabled
2. Plot initial positions
3. Run a few ticks
4. Plot final positions
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
    params.numticks = 10  # Run just a few ticks for demo
    params.numguys = 100  # Fewer agents for clearer visualization

    print("Creating simulation with plotting enabled...")
    model = Simulation.from_params(params, enable_plotting=True)

    # Plot initial positions
    print("Plotting initial positions...")
    model.plot_current_positions(
        title="Initial Agent Positions (Tick 0)",
        color_by_group=True,
        save_path="initial_positions.png"
    )

    # Run simulation for a few ticks
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

        # Plot every few ticks
        if tick % 5 == 0 or tick == params.numticks - 1:
            print(f"Plotting positions at tick {tick}...")
            model.plot_current_positions(
                title=f"Agent Positions at Tick {tick}",
                color_by_group=True,
                save_path=f"positions_tick_{tick:03d}.png"
            )

    model.storage.finalize(model.infobits)

    # Plot final positions
    print("Plotting final positions...")
    model.plot_current_positions(
        title=f"Final Agent Positions (Tick {params.numticks})",
        color_by_group=True,
        save_path="final_positions.png"
    )

    print("\nPlots saved successfully!")
    print("Generated files:")
    print("  - initial_positions.png")
    for tick in range(0, params.numticks, 5):
        print(f"  - positions_tick_{tick:03d}.png")
    print(f"  - positions_tick_{params.numticks-1:03d}.png")
    print("  - final_positions.png")

if __name__ == "__main__":
    main()
