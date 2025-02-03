from typing import Dict, List

import numpy as np

from src.network.network import HopfieldNetwork


class SimilaritiesLogger:
    def __init__(self):
        self.steps = []
        self.avg_similarity_history = []

    def log_step(self, networks: List[HopfieldNetwork], step: int) -> None:
        self.steps.append(step)
        sims, y = [], len(networks)
        for i in range(y):
            for j in range(i + 1, y):
                sims.append(networks[i].state_similarity(networks[j].state))
        self.avg_similarity_history.append(np.mean(sims))

    def get_data(self) -> Dict:
        return {"steps": self.steps, "avg_similarities": self.avg_similarity_history}
