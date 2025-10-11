"""
PositionPlotter for visualizing agent positions in opinion space.

This module provides plotting capabilities for the TripleFilterBubble simulation,
allowing visualization of agent (Guy) positions during or after simulation runs.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from models import Guy, GuyId
    from global_params import Params


@dataclass
class PositionPlotter:
    """
    Handles plotting of agent positions in 2D opinion space.

    This class is designed to be extensible for future plotting features
    such as animation, heatmaps, trajectory tracking, etc.

    Attributes:
        params: Simulation parameters for world bounds and styling
        fig: Matplotlib figure object
        ax: Matplotlib axes object
    """
    params: "Params"
    fig: Figure | None = None
    ax: Axes | None = None

    def setup_figure(self, figsize: tuple[float, float] = (10, 10)):
        """
        Initialize the matplotlib figure and axes.

        Args:
            figsize: Tuple of (width, height) in inches
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(-self.params.max_pxcor, self.params.max_pxcor)
        self.ax.set_ylim(-self.params.max_pxcor, self.params.max_pxcor)
        self.ax.set_xlabel("Opinion Dimension 1")
        self.ax.set_ylabel("Opinion Dimension 2")
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

    def plot_positions(
        self,
        guys: dict["GuyId", "Guy"],
        title: str = "Agent Positions in Opinion Space",
        color_by_group: bool = True,
        show_ids: bool = False,
        save_path: str | None = None
    ):
        """
        Plot current positions of all agents.

        Args:
            guys: Dictionary mapping GuyId to Guy objects
            title: Plot title
            color_by_group: If True, color agents by their group membership
            show_ids: If True, annotate each agent with their ID
            save_path: If provided, save the plot to this path instead of showing
        """
        if self.fig is None or self.ax is None:
            self.setup_figure()

        # Clear previous plot
        self.ax.clear()
        self.ax.set_xlim(-self.params.max_pxcor, self.params.max_pxcor)
        self.ax.set_ylim(-self.params.max_pxcor, self.params.max_pxcor)
        self.ax.set_xlabel("Opinion Dimension 1")
        self.ax.set_ylabel("Opinion Dimension 2")
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(title)

        # Extract positions and groups
        positions = np.array([guy.position for guy in guys.values()])

        if color_by_group:
            groups = np.array([guy.group for guy in guys.values()])
            scatter = self.ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=groups,
                cmap='tab10',
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )
            # Add colorbar
            cbar = self.fig.colorbar(scatter, ax=self.ax)
            cbar.set_label('Group')
        else:
            self.ax.scatter(
                positions[:, 0],
                positions[:, 1],
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )

        if show_ids:
            for guy_id, guy in guys.items():
                self.ax.annotate(
                    str(guy_id),
                    xy=guy.position,
                    fontsize=6,
                    alpha=0.7
                )

        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def close(self):
        """Close the figure to free memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
