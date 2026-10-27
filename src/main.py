from dataclasses import dataclass
from typing import Callable, Hashable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Аргументы внешнего поля: время и идентификатор узла.
ExternalField = Callable[[float, Hashable], float]


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


class SocialNetwork:
    """Граф, агенты и синхронное обновление."""

    def __init__(
        self,
        graph: nx.Graph,
        rule: SwitchingRule,
        external_field: ExternalField,
        initial_plus_probability: float = 0.5,
        seed: int | None = None,
    ):
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError("Ожидается простой неориентированный граф.")
        if nx.number_of_selfloops(graph):
            raise ValueError("Граф не должен содержать петель.")
        if len(graph) == 0:
            raise ValueError("Граф не должен быть пустым.")
        if not 0 <= initial_plus_probability <= 1:
            raise ValueError("Начальная вероятность должна быть в [0, 1].")

        self.graph = graph.copy()
        self.external_field = external_field
        self.rng = np.random.default_rng(seed)
        self.time = 0.0

        # Фиксированный порядок узлов для массивов результатов.
        self.nodes = tuple(self.graph.nodes)
        self.agents: dict[Hashable, Agent] = {}

        # Первый проход: создаём агентов и помещаем их в узлы.
        for node in self.nodes:
            state = (
                1 if self.rng.random() < initial_plus_probability else -1
            )
            agent = Agent(node, state, rule)
            self.agents[node] = agent
            self.graph.nodes[node]["agent"] = agent

        # Второй проход: устанавливаем ссылки на соседей.
        for node, agent in self.agents.items():
            agent.neighbors = [
                self.agents[neighbor]
                for neighbor in self.graph.neighbors(node)
            ]

    def states(self) -> np.ndarray:
        return np.array(
            [self.agents[node].state for node in self.nodes],
            dtype=np.int8,
        )

    def step(self, dt: float) -> int:
        """Один синхронный шаг; возвращает число переключений."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt должно быть конечным и > 0.")

        snapshot = {
            node: agent.state
            for node, agent in self.agents.items()
        }

        # Пока предложения вычисляются, реальные состояния не меняются.
        proposed = {
            node: agent.propose_state(
                snapshot,
                self.time,
                dt,
                self.external_field,
                self.rng,
            )
            for node, agent in self.agents.items()
        }

        switches = sum(
            proposed[node] != snapshot[node]
            for node in self.nodes
        )

        # Одновременное применение всех изменений.
        for node, state in proposed.items():
            self.agents[node].state = state

        self.time += dt
        return switches


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


def plot_results(result: SimulationResult):
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 9), constrained_layout=True
    )

    axes[0].plot(
        result.times, result.magnetization,
        label="Среднее мнение M(t)",
    )
    axes[0].plot(
        result.times, result.plus_fraction,
        label="Доля состояния +1",
    )
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_xlabel("Время")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(result.times[1:], result.switches)
    axes[1].set_ylabel("Переключений за шаг")
    axes[1].set_xlabel("Время")
    axes[1].grid(alpha=0.3)

    image = axes[2].imshow(
        result.states.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[2].set_xlabel("Номер шага")
    axes[2].set_ylabel("Индекс агента")
    fig.colorbar(image, ax=axes[2], ticks=[-1, 1], label="Состояние")

    plt.show()


if __name__ == "__main__":
    graph = nx.watts_strogatz_graph(
        n=200,
        k=8,
        p=0.1,
        seed=123,
    )

    rule = SwitchingRule(
        spontaneous=0.03,
        social=1.0,
        environmental=0.4,
    )

    def external_field(time, node_id):
        return 0.7 if time < 25.0 else -0.7

    network = SocialNetwork(
        graph=graph,
        rule=rule,
        external_field=external_field,
        initial_plus_probability=0.5,
        seed=42,
    )

    result = Simulator(network).run(steps=1000, dt=0.05)
    plot_results(result)
