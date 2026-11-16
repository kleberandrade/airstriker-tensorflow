"""Create a readable, truthful raw-frame/input comparison for project documentation."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from airstriker_tensorflow.preprocessing import FrameStack


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    frame = np.asarray(Image.open(args.frame).convert("RGB"))
    processed = np.asarray(Image.open(args.processed).convert("L"))
    expected = np.rint(FrameStack().reset(frame)[..., -1] * 255).astype(np.uint8)
    if not np.array_equal(processed, expected):
        raise ValueError("Processed image does not match the supplied raw frame")
    enlarged = np.repeat(np.repeat(processed, 12, axis=0), 12, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enlarged).save(args.output.with_name(args.output.stem + "-pixels.png"))
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.3), layout="constrained", gridspec_kw={"width_ratios": [320 / 224, 1]}
    )
    axes[0].imshow(frame, interpolation="nearest")
    axes[0].set_title("Actual game frame: 320 × 224 RGB", fontsize=14, pad=12)
    axes[1].imshow(processed, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1].set_title("Actual model channel: 32 × 32", fontsize=14, pad=12)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("From the game image to the model input", fontsize=18)
    fig.supxlabel(
        "Nearest-neighbor display · original pixel values · no contrast enhancement", fontsize=12
    )
    fig.savefig(args.output, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Verified matching frame/input; wrote comparison and exact 12× pixel enlargement.")


if __name__ == "__main__":
    main()
