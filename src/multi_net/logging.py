from collections import defaultdict
from typing import Dict

import numpy as np

from src.multi_net.ensemble import HopfieldEnsemble


class EnsembleLogger:
    def __init__(self):
        self.logs = defaultdict(list)

    def log_step(self, ensemble: HopfieldEnsemble, step: int) -> None:
        self.logs["steps"].append(step)
        sims, y, networks = [], ensemble.y, ensemble.networks
        for i in range(y):
            for j in range(i + 1, y):
                sims.append(networks[i].state_similarity(networks[j].state))
        self.logs["avg_similarity"].append(np.mean(sims))
        self.logs["unsat_with_replicas_interaction"].append(
            ensemble.num_unsatisfied_neurons_with_replicas_interaction()
        )

    def get_logs(self) -> Dict:
        return dict(self.logs)
