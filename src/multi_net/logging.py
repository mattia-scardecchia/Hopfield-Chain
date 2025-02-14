from collections import defaultdict
from typing import Dict

import numpy as np

from src.network.ensemble import HopfieldEnsemble


class EnsembleLogger:
    def __init__(self):
        self.logs = defaultdict(list)

    def log_step(self, ensemble: HopfieldEnsemble, step: int) -> None:
        self.logs["steps"].append(step)
        y, networks = ensemble.y, ensemble.networks
        sims = np.zeros((y, y))
        for i in range(y):
            for j in range(i + 1, y):
                sims[i, j] = networks[i].state_similarity(networks[j].state)
        self.logs["avg_similarity"].append(np.sum(sims) / (y * (y - 1) / 2))
        self.logs["avg_pairwise_similarity"].append(
            np.mean([sims[i, i + 1] for i in range(y - 1)])
        )
        # self.logs["avg_pairwise_similarity"].append(np.mean([]))
        self.logs["unsat_with_replicas_interaction"].append(
            ensemble.num_unsatisfied_neurons_with_replicas_interaction()
        )
        if ensemble.left_field is not None:
            self.logs["similarity_left_field"].append(
                [networks[i].state_similarity(ensemble.left_field) for i in range(y)]
            )
        if ensemble.right_field is not None:
            self.logs["similarity_right_field"].append(
                [networks[i].state_similarity(ensemble.right_field) for i in range(y)]
            )

    # TODO: this should be handled by each single HopfieldLogger!!
    # They should track initial states and converged fixed points.
    def log_fixed_point(self, ensemble: HopfieldEnsemble) -> None:
        for i, net in enumerate(ensemble.networks):
            self.logs[f"fixed_point_{i}"].append(net.state.copy())

    def get_logs(self) -> Dict:
        return dict(self.logs)

    def flush(self, keep_fixed_points: bool = False) -> None:
        for key in self.logs:
            if not keep_fixed_points or "fixed_point" not in key:
                self.logs[key] = []
