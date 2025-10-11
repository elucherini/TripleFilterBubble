"""
Measurement module for computing social distance metrics during simulation.

This module provides functions to compute:
- Mean link length (distance between guys and their infobits)
- Mean distance to info-sharers (distance between guys and the original posters)
- Mean distance to friends (distance between guys and their friends)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
import math

if TYPE_CHECKING:
    from main import Simulation


@dataclass
class MeasurementResults:
    """
    Stores measurement results indexed by tick.

    Each metric is stored as a dictionary mapping tick number to the computed value.
    If a metric cannot be computed (e.g., no infobits exist), the value is None.
    """
    mean_link_length: dict[int, float | None] = field(default_factory=dict)
    mean_infosharer_distance: dict[int, float | None] = field(default_factory=dict)
    mean_friend_distance: dict[int, float | None] = field(default_factory=dict)

    def add_measurement(
        self,
        tick: int,
        mean_link_length: float | None,
        mean_infosharer_distance: float | None,
        mean_friend_distance: float | None
    ):
        """Add a measurement for a specific tick."""
        self.mean_link_length[tick] = mean_link_length
        self.mean_infosharer_distance[tick] = mean_infosharer_distance
        self.mean_friend_distance[tick] = mean_friend_distance

    def __repr__(self) -> str:
        """Pretty-print measurement results."""
        lines = ["MeasurementResults:"]
        ticks = sorted(set(self.mean_link_length.keys()))
        if not ticks:
            lines.append("  No measurements recorded")
            return "\n".join(lines)

        lines.append(f"  {'Tick':<6} {'Link Length':<15} {'Info-Sharer':<15} {'Friend Dist':<15}")
        lines.append("  " + "-" * 56)
        for tick in ticks:
            ll = self.mean_link_length.get(tick)
            isd = self.mean_infosharer_distance.get(tick)
            fd = self.mean_friend_distance.get(tick)

            ll_str = f"{ll:.4f}" if ll is not None else "N/A"
            isd_str = f"{isd:.4f}" if isd is not None else "N/A"
            fd_str = f"{fd:.4f}" if fd is not None else "N/A"

            lines.append(f"  {tick:<6} {ll_str:<15} {isd_str:<15} {fd_str:<15}")

        return "\n".join(lines)


def compute_metrics(sim: "Simulation", tick: int) -> dict[str, float | None]:
    """
    Compute all social distance metrics for the current simulation state.

    Args:
        sim: The simulation instance
        tick: Current tick number (for logging/debugging)

    Returns:
        Dictionary with keys:
        - 'mean_link_length': Average distance between guys and their infobits
        - 'mean_infosharer_distance': Average distance between guys and original posters
        - 'mean_friend_distance': Average distance between guys and their friends

        Values are None if the metric cannot be computed (e.g., no data).
    """
    mean_link_length = _compute_mean_link_length(sim)
    mean_infosharer_distance = _compute_mean_infosharer_distance(sim)
    mean_friend_distance = _compute_mean_friend_distance(sim)

    return {
        'mean_link_length': mean_link_length,
        'mean_infosharer_distance': mean_infosharer_distance,
        'mean_friend_distance': mean_friend_distance
    }


def _compute_mean_link_length(sim: "Simulation") -> float | None:
    """
    Compute mean distance between guys and their infobits.

    For each guy, compute the distance to each of their infobits,
    then return the mean across all guy-infobit pairs.
    """
    distances = []

    for guy_id, guy in sim.guys.items():
        infobit_ids = sim.H.g2i.get(guy_id, set())
        if not infobit_ids:
            continue

        guy_pos = guy.position
        for infobit_id in infobit_ids:
            if infobit_id not in sim.infobits:
                continue
            infobit_pos = sim.infobits[infobit_id].position
            dist = sim.geo.norm_dist(guy_pos, infobit_pos)
            distances.append(dist)

    if not distances:
        return None

    return float(np.mean(distances))


def _compute_mean_infosharer_distance(sim: "Simulation") -> float | None:
    """
    Compute mean distance between guys and the guys who shared their infobits.

    For each guy, find the original poster (sharer) of each of their infobits,
    then compute the distance between the guy and the sharer.
    """
    distances = []

    for guy_id, guy in sim.guys.items():
        infobit_ids = sim.H.g2i.get(guy_id, set())
        if not infobit_ids:
            continue

        guy_pos = guy.position
        for infobit_id in infobit_ids:
            # Look up who shared this infobit to this guy
            sharer_id = sim.H.sharer.get((guy_id, infobit_id))
            if sharer_id is None:
                continue
            if sharer_id not in sim.guys:
                continue

            sharer_pos = sim.guys[sharer_id].position
            dist = sim.geo.norm_dist(guy_pos, sharer_pos)
            distances.append(dist)

    if not distances:
        return None

    return float(np.mean(distances))


def _compute_mean_friend_distance(sim: "Simulation") -> float | None:
    """
    Compute mean distance between guys and their friends.

    For each guy, compute the distance to each of their friends,
    then return the mean across all friend pairs.
    """
    distances = []

    # Use set to avoid counting each edge twice
    seen_edges = set()

    for guy_id, guy in sim.guys.items():
        friend_ids = sim.G.adj.get(guy_id, {}).keys()
        if not friend_ids:
            continue

        guy_pos = guy.position
        for friend_id in friend_ids:
            # Create canonical edge representation (smaller id first)
            edge = tuple(sorted([int(guy_id), int(friend_id)]))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)

            if friend_id not in sim.guys:
                continue

            friend_pos = sim.guys[friend_id].position
            dist = sim.geo.norm_dist(guy_pos, friend_pos)
            distances.append(dist)

    if not distances:
        return None

    return float(np.mean(distances))
