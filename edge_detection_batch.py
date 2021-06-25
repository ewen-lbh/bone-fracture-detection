#!/usr/bin/env python3
"""
Usage:
    edge_detection_batch.py TARGET START STOP STEP [--blur=KSIZES...] [--aperture=σ]

Options:
    --blur=KSIZES...    Kernel sizes for blurring. No blur if set to 0.
                        Multiple values to test different settings.
    --aperture=σ        σ (apertureSize) to use [default: 3]
"""
from docopt import docopt
from typing import Iterable, Union
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from rich.progress import Progress
from detect import contrast_of, detect_edges

# _, low, high, = sys.argv



def display_edges(
    fig_idx: int,
    title: str,
    filename: Path,
    low: int,
    high: int,
    σ: int = 3,
    blur: int = 0,
):
    _, edges = detect_edges(filename, low, high, σ=σ, blur=blur)
    plt.subplot(1, 2, fig_idx)
    plt.title(title)
    plt.imshow(edges, cmap="gray")


def batch_output_filename(target: Path, low: int, high: int, blur:int=0) -> Path:
    directory = (
        Path(__file__).parent / "edge-detection/batches" / target.with_suffix("")
    )
    directory.mkdir(parents=True, exist_ok=True)
    return (
        directory
        / f"blur={blur}-{low*high:04d}-low={low:03d}-high={high:03d}.png"
    )


def do_batch(
    image: Union[tuple[Path, Path], Path],
    blur_values: Iterable[int],
    high_values: Iterable[int],
    low_values: Iterable[int],
    σ: int = 3,
):
    """
    Does a batch of edge detection on image image with all combinations of provided low & high tresholds.
    - image: A tuple of paths (broken bone image, baseline or "healed" image) or a single path (broken bone only)
    - use_blur: whether to preprocess the image with blur
    - σ: sets aperture size.
    """
    if isinstance(image, tuple):
        broken, baseline = image
    else:
        broken, baseline = image, None
    with Progress() as progress_bar:
        task = progress_bar.add_task(
            "[blue]Processing...", total=len(list(high_values)) * len(list(low_values) * len(list(blur_values)))
        )

        for blur in blur_values:
            for high in high_values:
                for low in low_values:
                    plt.suptitle(
                        f"Seuils: haut: {high}, bas: {low}. σ: {σ}.\nPré-traitements: {'Flou: %d' % blur if blur else 'Aucun'}"
                        f"\n Contraste: {contrast_of(image)}"
                    )

                    plt.subplot(1, 2, 1)
                    plt.title("Image originale (cassé)")
                    plt.imshow(cv2.imread(str(broken), 0))

                    display_edges(2, "Cassé", broken, low, high, σ=σ, blur=blur)
                    if baseline:
                        display_edges(
                            3, "Sain", baseline, low, high, σ=σ, blur=blur
                        )

                    plt.savefig(
                        batch_output_filename(broken, low=low, high=high, blur=blur)
                    )
                    progress_bar.update(
                        task,
                        advance=1,
                        description=Path(
                            batch_output_filename(
                                broken, low=low, high=high, blur=blur
                            )
                        ).name,
                    )


if __name__ == "__main__":
    opts = docopt(__doc__)
    target = Path(opts["TARGET"])
    threshold_values = list(
        range(int(opts["START"]), int(opts["STOP"]), int(opts["STEP"]))
    )

    do_batch(
        # image=(target, target.parent / f"{target.with_suffix('').name}_baseline.png"),
        image=target,
        high_values=threshold_values,
        low_values=threshold_values,
        blur_values=list(map(int, opts['--blur'])),
        σ=int(opts["--aperture"]),
    )
