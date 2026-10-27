from typing import Hashable
import numpy as np

from SwitchingRule import SwitchingRule
# from SocialNetwork import ExternalField

from main import ExternalField

class Agent:
    """Двухуровневый агент без памяти о предыдущих состояниях."""

    def __init__(
        self,
        node_id: Hashable,
        state: int,
        rule: SwitchingRule,
    ):
        if state not in (-1, 1):
            raise ValueError("Состояние должно быть -1 или +1.")

        self.node_id = node_id
        self.state = state
        self.rule = rule
        self.neighbors: list["Agent"] = []

    def propose_state(
        self,
        snapshot: dict[Hashable, int],
        time: float,
        dt: float,
        external_field: ExternalField,
        rng: np.random.Generator,
    ) -> int:
        # Все агенты читают один и тот же снимок сети.
        state = snapshot[self.node_id]
        neighbor_states = [
            snapshot[neighbor.node_id]
            for neighbor in self.neighbors
        ]

        rate = self.rule.rate(
            state,
            neighbor_states,
            external_field(time, self.node_id),
        )

        # Численно устойчивый эквивалент 1 - exp(-rate * dt).
        probability = -np.expm1(-rate * dt)

        return -state if rng.random() < probability else state