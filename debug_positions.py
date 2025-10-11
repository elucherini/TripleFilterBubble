"""Debug script to compare positions between Python and NetLogo."""
import json
import numpy as np
from main import Simulation
from global_params import Params

# Load NetLogo reference
with open('tests/fixtures/netlogo_reference.json', 'r') as f:
    netlogo_ref = json.load(f)

# Run Python simulation
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
    run_dir="/tmp/debug_positions"
)

sim = Simulation.from_params(params)

# Check INITIAL positions
print("INITIAL POSITIONS (after from_params, before run):")
print("=" * 80)
for gid, guy in sorted(sim.guys.items()):
    print(f"Guy {gid}: Python={guy.position}")

print("\nNetLogo final positions (for reference):")
for gid in range(20):
    netlogo_pos = netlogo_ref['final_positions'][str(gid)]
    print(f"Guy {gid}: NetLogo={netlogo_pos}")

sim.run()

print("\n\nFINAL POSITIONS (after run):")
print("=" * 80)
for gid, guy in sorted(sim.guys.items()):
    netlogo_pos = netlogo_ref['final_positions'][str(gid)]
    diff = np.linalg.norm(guy.position - np.array(netlogo_pos))
    match = "✓" if diff < 1e-5 else "✗"
    print(f"Guy {gid}: {match} Python={guy.position}, NetLogo={netlogo_pos}, diff={diff:.10f}")
