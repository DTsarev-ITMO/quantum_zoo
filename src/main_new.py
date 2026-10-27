from typing import Callable, Hashable
import matplotlib.pyplot as plt
import networkx as nx



from Simulator import Simulator, SimulationResult
from SocialNetwork import SocialNetwork
from SwitchingRule import SwitchingRule

# Аргументы внешнего поля: время и идентификатор узла.
ExternalField = Callable[[float, Hashable], float]


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


if __name__ == '__main__':
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
