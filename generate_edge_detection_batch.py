#!/usr/bin/env python3
"""
Usage:
    edge_detection_batch.py TARGET START STOP STEP [--blur] [--aperture]

Options:
    --blur              Whether to pre-process with blur
    --aperture=σ        σ (apertureSize) to use [default: 3]
"""
from os import mkdir
from docopt import docopt
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from pathlib import Path
from shutil import rmtree
from rich import print
from rich.progress import Progress


# _, low, high, = sys.argv

def detect_edges(
    filename: Path, low: int, high: int, σ: int = 3, use_blur: bool = True
):
    σ, low, high = map(int, (σ, low, high))

    image = cv2.imread(str(filename))
    if use_blur:
        image = cv2.blur(image, (3, 3))
    edges = cv2.Canny(image, low, high, apertureSize=σ, L2gradient=True)
    return image, edges


def display_edges(
    fig_idx: int,
    title: str,
    filename: Path,
    low: int,
    high: int,
    σ: int = 3,
    use_blur: bool = True,
):
    _, edges = detect_edges(filename, low, high, σ=σ, use_blur=use_blur)
    plt.subplot(131 + fig_idx)
    plt.title(title)
    plt.imshow(edges, cmap="gray")


def batch_output_filename(low: int, high: int, use_blur: bool):
    return f"edge-detection/batches/{ 'via-blur' if use_blur else 'dry' }/{low*high:0{len(str(int(sys.argv[2])**2))}d} low={low:03d} high={high:03d}.png"

def do_batch(
    image: tuple[Path, Path],
    use_blur: bool,
    high_values: Iterable[int],
    low_values: Iterable[int],
    σ: int = 3,
):
    """
    Does a batch of edge detection on image image with all combinations of provided low & high tresholds.
    - image: A tuple of paths (broken bone image, baseline or "healed" image)
    - use_blur: whether to preprocess the image with blur
    - σ: sets aperture size.
    """
    preprocesses = ["flou"] if use_blur else []
    broken, baseline = image
    with Progress() as progress_bar:
        task = progress_bar.add_task("[blue]Processing...", total=len(list(high_values)) * len(list(low_values)))

        for high in high_values:
            for low in low_values:
                plt.suptitle(
                    f"Seuils: haut: {high}, bas: {low}. σ: {σ}.\nPré-traitements: {', '.join(preprocesses)}"
                )

                plt.subplot(130)
                plt.title("Image originale (cassé)")
                plt.imshow(cv2.imread(str(broken), 0))

                display_edges(1, "Cassé", broken, low, high, σ=σ, use_blur=use_blur)
                display_edges(2, "Sain", baseline, low, high, σ=σ, use_blur=use_blur)

                plt.savefig(
                    batch_output_filename(low=low, high=high, use_blur=use_blur)
                )
                progress_bar.update(
                    task,
                    advance=1,
                    description=batch_output_filename(
                        low=low, high=high, use_blur=use_blur
                    ),
                )


if __name__ == "__main__":
    opts = docopt(__doc__)    
    target = Path(opts['TARGET'])
    threshold_values = range(int(opts['START']), int(opts['STOP']), int(opts['STEP']))

    do_batch(
        image=(target.with_suffix("broken.png"), target.with_suffix("base.png")),
        high_values=threshold_values,
        low_values=threshold_values,
        use_blur=opts['--blur'],
        σ=int(opts['--aperture'])
    )
