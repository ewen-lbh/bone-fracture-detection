#!/usr/bin/env python3
"""
Usage:
    edge_detection_batch.py TARGET START STOP STEP [options]

Options:
    --blur              Whether to pre-process with blur
    --aperture=σ        σ (apertureSize) to use [default: 3]
"""
from docopt import docopt
from typing import Iterable, Union
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
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
    plt.subplot(1, 2, fig_idx)
    plt.title(title)
    plt.imshow(edges, cmap="gray")


def batch_output_filename(target: Path, low: int, high: int, use_blur: bool) -> Path:
    directory = Path(__file__).parent / "edge-detection/batches" / target.with_suffix('')
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"blur={'yes' if use_blur else 'no'}-{low*high:04d}-low={low:03d}-high={high:03d}.png"


def do_batch(
    image: Union[tuple[Path, Path], Path],
    use_blur: bool,
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
    preprocesses = ["flou"] if use_blur else []
    if isinstance(image, tuple):
        broken, baseline = image
    else:
        broken, baseline = image, None
    with Progress() as progress_bar:
        task = progress_bar.add_task(
            "[blue]Processing...", total=len(list(high_values)) * len(list(low_values))
        )

        for high in high_values:
            for low in low_values:
                plt.suptitle(
                    f"Seuils: haut: {high}, bas: {low}. σ: {σ}.\nPré-traitements: {', '.join(preprocesses) if preprocesses else 'aucun'}"
                )


                plt.subplot(1,2,1)
                plt.title("Image originale (cassé)")
                plt.imshow(cv2.imread(str(broken), 0))

                display_edges(2, "Cassé", broken, low, high, σ=σ, use_blur=use_blur)
                if baseline:
                    display_edges(3, "Sain", baseline, low, high, σ=σ, use_blur=use_blur)

                plt.savefig(
                    batch_output_filename(
                        broken, low=low, high=high, use_blur=use_blur
                    )
                )
                progress_bar.update(
                    task,
                    advance=1,
                    description=Path(batch_output_filename(
                        broken, low=low, high=high, use_blur=use_blur
                    )).name,
                )


if __name__ == "__main__":
    opts = docopt(__doc__)
    target = Path(opts["TARGET"])
    threshold_values = list(range(int(opts["START"]), int(opts["STOP"]), int(opts["STEP"])))

    do_batch(
        # image=(target, target.parent / f"{target.with_suffix('').name}_baseline.png"),
        image=target,
        high_values=threshold_values,
        low_values=threshold_values,
        use_blur=opts["--blur"],
        σ=int(opts["--aperture"]),
    )
