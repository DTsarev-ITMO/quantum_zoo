from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SwitchingRule:
    """Параметры и закон переключения."""
    spontaneous: float = 0.05
    social: float = 1.0
    environmental: float = 0.2

    def __post_init__(self):
        values = (self.spontaneous, self.social, self.environmental)
        if any(not np.isfinite(v) or v < 0 for v in values):
            raise ValueError("Интенсивности должны быть конечными и >= 0.")

    def rate(
        self,
        state: int,
        neighbor_states: list[int],
        field: float,
    ) -> float:
        if not np.isfinite(field) or not -1 <= field <= 1:
            raise ValueError("Внешнее поле должно принадлежать [-1, 1].")

        social_rate = 0.0
        if neighbor_states:
            mean_opinion = float(np.mean(neighbor_states))
            social_rate = self.social * (1 - state * mean_opinion) / 2

        environment_rate = (
            self.environmental * (1 - state * field) / 2
        )

        return self.spontaneous + social_rate + environment_rate