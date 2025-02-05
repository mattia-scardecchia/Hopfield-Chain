from typing import List

import numpy as np

from src.network.network import HopfieldNetwork

# TODO: maintain a baricenter with state components given by a majority vote of replicas?


class HopfieldEnsemble:
    def __init__(self, networks: List[HopfieldNetwork], k: float, chained: bool):
        self.networks = networks
        self.y = len(networks)
        self.k = k
        self.N = networks[0].N

        self.chained = chained
        self.neighbors = [self._get_neighbors(i) for i in range(self.y)]
        self.scale = 2 if chained else (self.y - 1)

    def _get_neighbors(self, replica_idx: int) -> List[int]:
        if self.chained:
            if replica_idx == 0:
                return [1]
            elif replica_idx == self.y - 1:
                return [self.y - 2]
            else:
                return [replica_idx - 1, replica_idx + 1]
        else:
            return list(range(0, replica_idx)) + list(range(replica_idx + 1, self.y))

    def local_field(self, replica_idx: int, neuron_idx: int) -> float:
        internal_field = self.networks[replica_idx].local_field(neuron_idx)
        if self.y == 1:
            return internal_field

        interaction_field = sum(
            [self.networks[i].state[neuron_idx] for i in self.neighbors[replica_idx]]
        )
        return internal_field + self.k * interaction_field / self.scale

    def num_unsatisfied_neurons_with_replicas_interaction(self) -> List[int]:
        nums = []
        for replica_idx, network in enumerate(self.networks):
            num = 0
            for neuron_idx in range(self.N):
                h = self.local_field(replica_idx, neuron_idx)
                if np.sign(h) != network.state[neuron_idx]:
                    num += 1
            nums.append(num)
        return nums

    def check_convergence(self) -> bool:
        return self.num_unsatisfied_neurons_with_replicas_interaction() == [0] * self.y
