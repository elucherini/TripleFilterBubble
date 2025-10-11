"""Debug script to compare edges between Python and NetLogo."""
import json
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
    run_dir="/tmp/debug_edges"
)

sim = Simulation.from_params(params)
sim.run()

# Get edges from both
python_edges = set((int(u), int(v)) if u < v else (int(v), int(u)) for u, v in sim.G.edges())
netlogo_edges = set((u, v) if u < v else (v, u) for u, v in netlogo_ref['final_edges'])

print(f"Python edges: {len(python_edges)}")
print(f"NetLogo edges: {len(netlogo_edges)}")
print()

# Find differences
only_in_python = python_edges - netlogo_edges
only_in_netlogo = netlogo_edges - python_edges

print(f"Edges only in Python ({len(only_in_python)}):")
for edge in sorted(only_in_python):
    print(f"  {edge}")

print(f"\nEdges only in NetLogo ({len(only_in_netlogo)}):")
for edge in sorted(only_in_netlogo):
    print(f"  {edge}")

print(f"\ninfobits_created: Python={sim.infobits_created}, NetLogo={netlogo_ref['num_infobits_final']}")
