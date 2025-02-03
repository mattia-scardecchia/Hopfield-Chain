import numpy as np
from matplotlib import pyplot as plt


class ReplicatedPlotter:
    def __init__(self, loggers, similarities_logger):
        self.loggers = loggers
        self.similarities_logger = similarities_logger

        self.y = len(loggers)
        keys = loggers[0].get_data().keys()
        self.data = {
            key: np.array([logger.get_data()[key] for logger in loggers])
            for key in keys
        }

    def _plot_metric_individually(
        self, ax: plt.Axes, key_name: str, ylabel: str, title: str
    ):
        data = self.data[key_name]
        steps = self.data["steps"][0]
        for i in range(self.y):
            ax.plot(steps, data[i], label=f"Replica {i}")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    def _plot_replicas_similarity(self, ax: plt.Axes):
        steps = self.similarities_logger.get_data()["steps"]
        avg_similarities = self.similarities_logger.get_data()["avg_similarities"]
        ax.plot(steps, avg_similarities, color="blue")
        ax.set_ylabel("Average Similarity")
        ax.set_xlabel("Step")
        ax.set_title("Average Similarity Between Replicas")
        ax.grid(True, linestyle="--", alpha=0.6)

    def plot_all(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        self._plot_metric_individually(
            axes[0, 0],
            "unsatisfied",
            "Unsat Neurons",
            "Unsat Neurons vs. Step",
        )
        self._plot_metric_individually(
            axes[1, 1], "similarities", "Similarity", "Similarity vs. Step"
        )
        self._plot_replicas_similarity(axes[0, 1])
        self._plot_metric_individually(
            axes[1, 0], "magnetizations", "Magnetization", "Magnetization vs. Step"
        )
        # self._plot_metric_individually(
        #     axes[0, 1], "energies", "Energy", "Energy vs. Step"
        # )
        plt.tight_layout()
        return fig
