from dataclasses import dataclass
import numpy as np

from SocialNetwork import SocialNetwork


@dataclass
class SimulationResult:
    nodes: tuple
    times: np.ndarray
    states: np.ndarray
    switches: np.ndarray

    @property
    def magnetization(self) -> np.ndarray:
        return self.states.mean(axis=1)

    @property
    def plus_fraction(self) -> np.ndarray:
        return (1 + self.magnetization) / 2


class Simulator:
    def __init__(self, network: SocialNetwork):
        self.network = network

    def run(self, steps: int, dt: float) -> SimulationResult:
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps должно быть целым и >= 0.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt должно быть конечным и > 0.")

        n = len(self.network.nodes)
        history = np.empty((steps + 1, n), dtype=np.int8)
        times = np.empty(steps + 1)
        switches = np.empty(steps, dtype=int)

        history[0] = self.network.states()
        times[0] = self.network.time

        for k in range(steps):
            switches[k] = self.network.step(dt)
            history[k + 1] = self.network.states()
            times[k + 1] = self.network.time

        return SimulationResult(
            nodes=self.network.nodes,
            times=times,
            states=history,
            switches=switches,
        )