# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python implementation of the TripleFilterBubble agent-based model from the paper "The triple filter bubble: Using agent-based modelling to test a meta-theoretical framework for the emergence of filter bubbles and echo chambers" (Geschke, Lorenz, Holtz 2018).

The codebase is a performance-optimized Python port of the original NetLogo simulation model (`TripleFilterBubble.nlogo`). It simulates social networks where agents ("guys") share and integrate information ("infobits") based on proximity in opinion space.

## Commands

### Running the simulation
```bash
python src/main.py
```

The simulation runs with parameters defined in `src/global_params.py` (default: 100 ticks, 500 guys, seed 42). Results are stored in compressed format in the `data/` directory.

### Profiling
The main script includes built-in cProfile support. Profile results are saved as `.prof` files (e.g., `posting.prof`). To view:
```bash
python -m snakeviz posting.prof
```

### Dependencies
The project uses `uv` for dependency management:
```bash
uv sync
```

Or with pip:
```bash
pip install -r pyproject.toml
```

## Architecture

### Core simulation flow ([src/main.py](src/main.py))

The `Simulation` class orchestrates the model:

1. **Initialization** (`from_params`): Creates agents, builds social network, initializes bipartite graph
2. **Main loop** (`run`): For each tick:
   - `new_infobits`: Agents encounter new information (central, individual, or selective modes)
   - `post_infobits`: Agents share information with friends
   - `birth_death`: Optional agent replacement
   - `refriend`: Optional network rewiring based on opinion distance
   - `update_infobits`: Update popularity metrics
   - `visualize`: Compute position fluctuation metrics
   - Storage: Write compressed simulation state

### Key data structures

- **Guy** ([src/models.py:13](src/models.py#L13)): Agent with 2D opinion position, group membership, and incremental position tracking (`inf_sum`, `inf_count` for efficient mean updates)
- **Infobit** ([src/models.py:55](src/models.py#L55)): Information piece with 2D position and popularity
- **BiAdj** ([src/models.py:71](src/models.py#L71)): Bidirectional adjacency structure for guy↔infobit links with optional callbacks for event logging
- **Simulation** ([src/main.py:16](src/main.py#L16)): Main container holding:
  - `guys`: Dict of agents
  - `G`: NetworkX graph of friendships
  - `H`: Bipartite graph of who knows what
  - `geo`: Fast geometry calculations
  - `storage`: Compressed output writer
  - `_grid`: Optional spatial index for "select close infobits" mode

### Performance optimizations

This codebase is heavily optimized for speed. Key patterns:

1. **Vectorized distance calculations** ([src/main.py:125-129](src/main.py#L125-L129), [src/main.py:214-218](src/main.py#L214-L218)): Uses NumPy broadcasting to compute distances for all agents at once
2. **Precomputed geometry** ([src/utils.py:12](src/utils.py#L12) `FastGeo`): Powers and inverse norms are cached
3. **Spatial grid** ([src/utils.py:273](src/utils.py#L273) `SpatialGrid`): 3×3 neighbor lookup for close infobit selection
4. **Incremental position updates** ([src/models.py:19-20](src/models.py#L19-L20)): Agents track `inf_sum` and `inf_count` to compute mean position without reductions
5. **Slots** ([src/models.py:12](src/models.py#L12), [src/models.py:54](src/models.py#L54), [src/models.py:70](src/models.py#L70)): All dataclasses use `slots=True` for memory efficiency
6. **Static graph optimization** ([src/main.py:289-291](src/main.py#L289-L291)): If `refriend_probability=0`, the friendship graph is written once and reused

### Storage system ([src/utils.py:38](src/utils.py#L38))

`FastStorage` writes compressed simulation data:

- **Guy positions**: Memory-mapped int16 array (T×N×2), quantized to 16-bit integers
- **Friendship graph**: Packed bitset per tick (upper triangle only)
- **Bipartite events**: Append-only binary log of (guy_idx, infobit_idx, add/remove) events
- **Infobits**: Final positions saved at end in event-log index order
- **Compression**: All outputs are zstd-compressed level 3 after simulation completes

The storage callbacks are attached to `BiAdj` via `attach_biadj_callbacks` to log all info-link changes.

### Parameter modes

**new_info_mode** ([src/global_params.py:28](src/global_params.py#L28)):
- `"central"`: One infobit per tick shared by all nearby agents
- `"individual"`: Each agent creates their own infobit
- `"select close infobits"`: Agents try to integrate existing nearby infobits (uses spatial grid)
- `"select distant infobits"`: Agents try to integrate existing distant infobits (random sampling)

**Network dynamics**:
- `refriend_probability > 0`: Agents rewire friendships away from distant opinions toward friends-of-friends
- `birth_death_probability > 0`: Agents are randomly replaced with fresh agents

## Testing

### NetLogo regression test ([tests/test_golden_regression.py](tests/test_golden_regression.py))

**Critical:** This test verifies that the Python implementation produces identical results to the original NetLogo model. It compares against reference data in [tests/fixtures/netlogo_reference.json](tests/fixtures/netlogo_reference.json).

**Purpose:** Ensure behavioral parity between Python and NetLogo implementations. This test is the single source of truth for correctness.

**What it validates:**
- Total infobits created (cumulative counter matching NetLogo's `infobits-created`)
- Final friendship network edges (exact edge list)
- Total info-links between guys and infobits
- Final guy positions (within floating point tolerance)

**When to run:**
- After any changes to core simulation logic
- Before committing performance optimizations
- When modifying guy movement, infobit creation, or network dynamics

Run with: `pytest tests/test_golden_regression.py -m e2e`

**Important:** If this test fails, the Python implementation has diverged from NetLogo. Do not proceed with changes until parity is restored.

## Development notes

- The codebase prioritizes performance over abstraction. Many loops are vectorized or use NumPy operations directly.
- Git history shows multiple optimization passes (see commits with "optim", "refriend_optimized", "slots", "grid" in `.prof` files).
- Profile files (`.prof`) in the root are working artifacts and should not be committed.
- The `data/` directory contains simulation outputs and is gitignored.
- When modifying core simulation logic, verify parity with NetLogo model behaviors (e.g., torus wrapping, friend-of-friend selection).
