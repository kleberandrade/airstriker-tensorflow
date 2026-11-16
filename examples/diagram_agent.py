"""Draw the documented default Airstriker DQN data flow as PNG and SVG."""

from pathlib import Path

from airstriker_tensorflow.agent import DQNConfig


def box(ax, x, y, width, height, text, network=False):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012",
        linewidth=1.1,
        edgecolor="#374151",
        facecolor="#e7eff7" if network else "#f3f4f6",
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        color="#111827",
        linespacing=1.4,
    )


def arrow(ax, start, end, label="", position=None, dashed=False):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.3,
            "color": "#374151",
            "linestyle": "--" if dashed else "-",
        },
    )
    if label:
        x, y = position or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
        )


def main():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "airstriker-dqn-flow"
    import matplotlib.pyplot as plt

    config = DQNConfig()
    fig, axes = plt.subplots(2, 1, figsize=(9, 10.5), layout="constrained")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    act, learn = axes
    act.set_title("1. Play: choose an action from the current observation", fontsize=15, pad=15)
    box(act, 0.04, 0.66, 0.36, 0.21, "Airstriker + Stable Retro\nGame state and rewards")
    box(act, 0.60, 0.66, 0.36, 0.21, "Pillow + NumPy\nRGB → grayscale → 32 × 32\nStack four frames")
    box(
        act,
        0.60,
        0.22,
        0.36,
        0.21,
        "TensorFlow / Keras\nOnline network Q(s, a)\n126 action values",
        True,
    )
    box(
        act, 0.04, 0.22, 0.36, 0.21, "Action selection\nTraining: epsilon = 0.1\nEvaluation: greedy"
    )
    arrow(act, (0.42, 0.76), (0.58, 0.76), "RGB frame", (0.5, 0.82))
    arrow(act, (0.78, 0.64), (0.78, 0.45), "32 × 32 × 4", (0.81, 0.54))
    arrow(act, (0.58, 0.32), (0.42, 0.32), "Q-values", (0.5, 0.38))
    arrow(act, (0.22, 0.45), (0.22, 0.64), "controller\nbuttons", (0.16, 0.54))
    act.text(
        0.5,
        0.06,
        "After env.step: store (state, action, reward, next state, terminal).",
        ha="center",
        fontsize=10,
    )

    learn.set_title(
        "2. Learn: sample stored transitions and update online weights", fontsize=15, pad=15
    )
    box(
        learn,
        0.04,
        0.71,
        0.38,
        0.19,
        f"Replay buffer\nUp to {config.replay_capacity:,} transitions\nCopied uint8 pixels",
    )
    box(
        learn,
        0.61,
        0.71,
        0.35,
        0.19,
        "TensorFlow / Keras\nFrozen target network\nNext-state Q_target(s′)",
        True,
    )
    box(
        learn,
        0.04,
        0.24,
        0.38,
        0.23,
        "Targets + Huber loss\nFrozen next-state estimates\nOnline sampled-action values",
    )
    box(
        learn,
        0.61,
        0.24,
        0.35,
        0.23,
        "TensorFlow / Keras\nAdam: learning rate 0.001\nUpdate ONLINE weights only",
        True,
    )
    arrow(learn, (0.44, 0.80), (0.59, 0.80), "sampled s′", (0.51, 0.86))
    arrow(
        learn,
        (0.23, 0.69),
        (0.23, 0.49),
        f"Uniform batch {config.batch_size}\nStart at {config.learning_starts} stored",
        (0.23, 0.59),
    )
    arrow(learn, (0.65, 0.69), (0.40, 0.49), "max next Q", (0.5, 0.60))
    arrow(learn, (0.44, 0.35), (0.59, 0.35), "minimize\nloss", (0.51, 0.40))
    arrow(
        learn,
        (0.82, 0.49),
        (0.82, 0.69),
        f"Copy every {config.target_sync_interval}\noptimizer updates",
        (0.82, 0.59),
        True,
    )
    learn.text(
        0.5,
        0.10,
        "y = reward + 0.99 × (1 − terminal) × max Q_target(next state)",
        ha="center",
        fontsize=11,
    )
    learn.text(
        0.5,
        0.02,
        "Evaluation uses the online checkpoint, with no replay writes or updates.",
        ha="center",
        fontsize=10,
    )
    fig.suptitle("How TensorFlow learns to act in Airstriker", fontsize=17)
    output = Path("images/airstriker-dqn-flow")
    fig.savefig(output.with_suffix(".png"), dpi=140, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
