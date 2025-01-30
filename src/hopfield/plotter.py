from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, List
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class HopfieldPlotter:
    """
    Plots the data gathered by a Logger from the HopfieldSimulation.
    Reuses a single internal plotting method for each metric,
    allowing easy extension if you add new metrics.
    """

    def __init__(self, data: Dict) -> None:
        """
        Initializes the plotter with the data to be plotted.

        Parameters
        ----------
        data : Dict[str, List]
            A dictionary containing logged data. Expected keys include:
            "energies" -> List[float]
            "magnetizations" -> List[float]
            "log_interval" -> int (interval at which data was logged)
            "states" -> List[np.ndarray] (optional, if needed)
        """
        self.data = data
        self.energies = self.data.get("energies", [])
        self.magnetizations = self.data.get("magnetizations", [])
        self.similarities = self.data.get("similarities", [])
        self.log_interval = self.data.get("log_interval", 1)
        self.x_values = np.arange(len(self.energies)) * self.log_interval

    def _plot_metric(
        self, ax: plt.Axes, y_values: List[float], color: str, ylabel: str, title: str
    ) -> None:
        """
        Internal helper method to plot a single metric on a given axis.

        Parameters
        ----------
        ax : plt.Axes
            The axis on which to plot the data.
        y_values : List[float]
            The metric values to plot.
        color : str
            The color to use for the plot line.
        ylabel : str
            The label for the y-axis.
        title : str
            The title of the subplot.
        """
        ax.plot(self.x_values, y_values, color=color)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6)

    def plot_energies(self, title: str = "Energy vs. Step") -> None:
        """
        Creates a figure and plots the energy over simulation steps,
        taking into account the log interval.

        Parameters
        ----------
        title : str
            Title for the energy plot.
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        self._plot_metric(ax, self.energies, "blue", "Energy", title)
        ax.set_xlabel("Step")
        plt.tight_layout()
        plt.show()

    def plot_magnetization(self, title: str = "Magnetization vs. Step") -> None:
        """
        Creates a figure and plots the magnetization over simulation steps,
        taking into account the log interval.

        Parameters
        ----------
        title : str
            Title for the magnetization plot.
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        self._plot_metric(ax, self.magnetizations, "red", "Magnetization", title)
        ax.set_xlabel("Step")
        plt.tight_layout()
        plt.show()

    def plot_all(self):
        """
        Creates a figure with two subplots: one for energy and one for magnetization.
        Reuses the _plot_metric method on each axis.
        """
        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(6, 6), sharex=True, squeeze=False
        )

        self._plot_metric(
            ax=axes[0, 0],
            y_values=self.energies,
            color="blue",
            ylabel="Energy",
            title="Energy vs. Step",
        )
        self._plot_metric(
            ax=axes[1, 0],
            y_values=self.magnetizations,
            color="red",
            ylabel="Magnetization",
            title="Magnetization vs. Step",
        )
        self._plot_metric(
            ax=axes[0, 1],
            y_values=self.similarities,
            color="green",
            ylabel="Similarity",
            title="Similarity vs. Step",
        )

        axes[1, 1].axis("off")
        plt.tight_layout()
        return fig
