from collections import defaultdict
from typing import List, Optional

import numpy as np

from src.network.network import HopfieldNetwork

# TODO: maintain a baricenter with state components given by a majority vote of replicas?


class HopfieldEnsemble:
    def __init__(
        self,
        networks: List[HopfieldNetwork],
        k: float,
        chained: bool,
        left_field: Optional[np.ndarray] = None,
        right_field: Optional[np.ndarray] = None,
        h: float = 0.0,
    ):
        assert len(networks) > 1, "For a single network, use HopfieldNetwork instead"

        self.networks = networks
        self.y = len(networks)
        self.k = k
        self.N = networks[0].N

        self.chained = chained
        self.neighbors = [self._get_neighbors(i) for i in range(self.y)]
        self.scale = 2 if chained else (self.y - 1)

        if not chained:
            assert left_field is None and right_field is None
        self.left_field = left_field
        self.right_field = right_field
        self.h = h

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
        interaction_field = sum(
            [self.networks[i].state[neuron_idx] for i in self.neighbors[replica_idx]]
        )
        total_field = internal_field + self.k * interaction_field / self.scale

        if replica_idx == 0 and self.left_field is not None:
            total_field += self.h * self.left_field[neuron_idx]
        if replica_idx == self.y - 1 and self.right_field is not None:
            total_field += self.h * self.right_field[neuron_idx]

        return total_field

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

    def get_local_field_breakdown(self, replica_idx: int, neuron_idx: int) -> dict:
        internal_field = self.networks[replica_idx].local_field(neuron_idx)
        interaction_field = (
            sum(
                [
                    self.networks[i].state[neuron_idx]
                    for i in self.neighbors[replica_idx]
                ]
            )
            / self.scale
        )
        external_field = 0.0
        if replica_idx == 0 and self.left_field is not None:
            external_field = self.left_field[neuron_idx]
        if replica_idx == self.y - 1 and self.right_field is not None:
            external_field = self.right_field[neuron_idx]

        return {
            "internal": internal_field,
            "interaction": interaction_field,
            "external": external_field,
            "internal_weighted": internal_field,
            "interaction_weighted": self.k * interaction_field,
            "external_weighted": self.h * external_field,
            "total": internal_field
            + self.k * interaction_field
            + self.h * external_field,
        }

    def local_field_from_external_field_pov(
        self, component_idx: int, right: bool = True
    ) -> float:
        """
        For inference with a trained classifer.
        """
        net_idx = self.y - 1 if right else 0
        return self.h * self.networks[net_idx].state[component_idx]

    def num_unsatisfied_external_field_components(self, right: bool = True) -> int:
        """
        For inference with a trained classifer.
        """
        field = self.right_field if right else self.left_field
        num = 0
        for component_idx in range(self.N):
            h = self.local_field_from_external_field_pov(component_idx, right)
            if np.sign(h) != field[component_idx]:
                num += 1
        return num

    def external_field_is_stable(self, right: bool = True):
        """
        For inference with a trained classifer.
        """
        return self.num_unsatisfied_external_field_components(right) == 0


def collect_field_breakdowns(ensemble: HopfieldEnsemble, n: int):
    """
    For each layer, sample n neurons and get their local field breakdown. Return a dictionary
    with the breakdowns for each layer: replica_idx -> field_type -> list of values.
    """

    breakdowns = defaultdict(lambda: defaultdict(list))
    for replica_idx in range(ensemble.y):
        idxs = np.random.choice(ensemble.N, n) if n != -1 else range(ensemble.N)
        for neuron_idx in idxs:
            breakdown = ensemble.get_local_field_breakdown(replica_idx, neuron_idx)
            for key, value in breakdown.items():
                breakdowns[replica_idx][key].append(value)
    return breakdowns
