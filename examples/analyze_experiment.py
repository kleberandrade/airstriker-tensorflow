"""Audit raw/compressed traces and create reproducible static experiment figures.

Install the optional report dependencies: python -m pip install -e '.[report]'.
"""

import argparse
import json
from pathlib import Path

from airstriker_tensorflow.analysis import (
    analyze_experiment,
    compare_evaluations,
    compare_training_budgets,
)


def evaluation_figure(summary, output):
    import matplotlib.pyplot as plt

    phases = ["random", "untrained", "trained"]
    labels = ["Random", "Untrained network", "Checkpoint network"]
    colors = ["#707070", "#3b6d9a", "#cb772f"]
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for index, (phase, color) in enumerate(zip(phases, colors, strict=True)):
        group = summary["groups"][phase]
        ax.bar(index, group["mean_reward"], color=color, alpha=0.65, width=0.55)
        offsets = [
            index + (n - (len(group["rewards"]) - 1) / 2) * 0.11
            for n in range(len(group["rewards"]))
        ]
        ax.scatter(offsets, group["rewards"], color="black", s=35, zorder=3)
    budget = summary["configuration"]["evaluate_steps"]
    ax.set_xticks(range(len(phases)), labels)
    ax.set_ylabel(f"Total evaluation reward / {budget:,} steps")
    ax.set_title("Airstriker: equal-budget evaluation\nBars = mean; dots = individual seeds")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=-max(1.0, max(max(summary["groups"][p]["rewards"]) for p in phases) * 0.05))
    save_figure(fig, output / "evaluation-comparison")
    plt.close(fig)


def training_figure(summary, output):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for row in summary["runs"]:
        if row["phase"] == "train":
            windows = row["trace_statistics"]["reward_windows"]
            ax.plot(
                [w["end"] for w in windows],
                [w["reward"] for w in windows],
                marker="o",
                label=f"Seed {row['seed']}",
            )
    ax.set_xlabel("Training transitions (end of window)")
    ax.set_ylabel(f"Reward in each {summary['window_steps']:,}-step window")
    ax.set_title(
        "Airstriker: training reward windows\nThese are training rewards, not evaluation scores"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    save_figure(fig, output / "training-rewards")
    plt.close(fig)


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=140, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": None}, bbox_inches="tight")


def policy_comparison_figure(comparison, phases, labels, output, name, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for index, phase in enumerate(phases):
        rewards = [row[phase] for row in comparison["per_seed"]]
        ax.bar(index, sum(rewards) / len(rewards), width=0.55, color="steelblue", alpha=0.65)
        offsets = [index + (n - (len(rewards) - 1) / 2) * 0.11 for n in range(len(rewards))]
        ax.scatter(offsets, rewards, color="black", zorder=3)
    ax.set_xticks(range(len(phases)), labels)
    ax.set_ylabel(f"Evaluation reward / {comparison['protocol']['evaluate_steps']:,} steps")
    ax.set_title(title + "\nBars = mean; dots = individual seeds")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    maximum = max(max(row[p] for p in phases) for row in comparison["per_seed"])
    ax.set_ylim(bottom=-max(1.0, maximum * 0.05))
    save_figure(fig, output / name)
    plt.close(fig)


def algorithm_figure(comparison, output):
    policy_comparison_figure(
        comparison,
        ["random", "untrained", "online", "dqn"],
        ["Random", "Untrained", "Online Q-learning", "Replay + target"],
        output,
        "algorithm-comparison",
        "Airstriker: same-seed algorithm comparison",
    )


def budget_figure(comparison, output):
    steps = comparison["training_steps"]
    labels = ["Random", "Untrained", f"DQN {steps['previous']:,}", f"DQN {steps['current']:,}"]
    policy_comparison_figure(
        comparison,
        ["random", "untrained", "previous", "current"],
        labels,
        output,
        "training-budget-comparison",
        "Airstriker: training-budget comparison",
    )


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "airstriker-tensorflow"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument(
        "--previous",
        type=Path,
        help="Earlier experiment to compare; use training-budget mode when only train steps differ",
    )
    parser.add_argument(
        "--comparison",
        choices=["algorithm", "training-budget"],
        default="algorithm",
        help="algorithm requires equal train budgets; training-budget requires matching DQN settings",
    )
    args = parser.parse_args(argv)
    summary = analyze_experiment(args.source, args.window)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    evaluation_figure(summary, args.output)
    training_figure(summary, args.output)
    if args.previous:
        previous = analyze_experiment(args.previous, args.window)
        comparator = (
            compare_training_budgets
            if args.comparison == "training-budget"
            else compare_evaluations
        )
        comparison = comparator(summary, previous)
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        plot = budget_figure if args.comparison == "training-budget" else algorithm_figure
        plot(comparison, args.output)
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
