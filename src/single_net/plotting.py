from typing import Dict, List

from matplotlib import pyplot as plt


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
        self.log_steps = self.data.get("steps", [])
        self.unsatisfied = self.data.get("unsat", [])
        self.energies = self.data.get("pseudo_energy", [])
        self.magnetizations = self.data.get("magnetization", [])
        self.similarities = self.data.get("ref_state_similarity", [])

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
        ax.plot(self.log_steps, y_values, color=color)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6)

    def plot_all(self):
        """
        Creates a figure with two subplots: one for energy and one for magnetization.
        Reuses the _plot_metric method on each axis.
        """
        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(12, 8), sharex=True, squeeze=False
        )
        self._plot_metric(
            ax=axes[0, 0],
            y_values=self.unsatisfied,
            color="black",
            ylabel="Unsat neurons",
            title="Unsat vs. Step",
        )
        axes[0, 0].set_title("Unsat vs. Step (final: {})".format(self.unsatisfied[-1]))
        self._plot_metric(
            ax=axes[0, 1],
            y_values=self.magnetizations,
            color="red",
            ylabel="Magnetization",
            title="Magnetization vs. Step",
        )
        self._plot_metric(
            ax=axes[1, 0],
            y_values=self.similarities,
            color="green",
            ylabel="Similarity",
            title="Similarity vs. Step",
        )
        axes[1, 0].axhline(y=0.5, color="grey", linestyle="--", alpha=0.6)
        self._plot_metric(
            ax=axes[1, 1],
            y_values=self.energies,
            color="blue",
            ylabel="Pseudo-Energy",
            title="Pseudo-Energy vs. Step",
        )
        plt.tight_layout()
        return fig
