from typing import Callable, Hashable
import networkx as nx
import numpy as np

from SwitchingRule import SwitchingRule
from Agent import Agent

from main  import ExternalField


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