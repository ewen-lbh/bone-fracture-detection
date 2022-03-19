#!/usr/bin/env python3
"""
Usage:
    edge_detection_batch.py TARGET START STOP STEP [--blur=KSIZES...] [--contrast=CONTRAST...] [--aperture=σ]

Options:
    --blur=KSIZES...        Kernel sizes for blurring. No blur if set to 0.
                            Multiple values to test different settings.  [default: 0]
    --contrast=CONTRAST...  Contrast multiplier. None applied if set to 1.
                            Multiple values to test different settings.  [default: 1]
    --aperture=σ            σ (apertureSize) to use [default: 3]
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import cv2
import matplotlib.pyplot as plt
from deco import concurrent, synchronized
from docopt import docopt

from detect import brightness_of, contrast_of, detect_edges, grayscale_of

warnings.filterwarnings("ignore")

# _, low, high, = sys.argv


@dataclass
class Progress:
    total: int = 1
    done: int = 0


def display_edges(
    fig_idx: int,
    title: str,
    image,
    low: int,
    high: int,
    σ: int = 3,
    blur: int = 0,
):
    _, edges = detect_edges(image, low, high, σ=σ, blur=blur)
    plt.subplot(1, 2, fig_idx)
    global current_edges_brightness
    current_edges_brightness = brightness_of(edges)
    plt.title(brightness_of(edges))
    plt.imshow(edges, cmap="gray")


def batch_output_filename(
    target: Path, low: int, high: int, blur: int = 0, boost_contrast: int = 1
) -> Path:
    directory = (
        Path(__file__).parent / "edge-detection/batches" / target.with_suffix("")
    )
    directory.mkdir(parents=True, exist_ok=True)
    return (
        directory
        / f"blur={blur}-mult={boost_contrast}-{low+high:04d}-low={low:03d}-high={high:03d}.png"
    )


@concurrent
def do_one(image_path: Path, image, low, high, σ, blur, progress, boost_contrast):
    if blur:
        image = cv2.blur(image, (blur, blur))

    if boost_contrast != 1:
        image *= boost_contrast

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(brightness_of(image))

    display_edges(2, "Cassé", image, low, high, σ=σ, blur=0)

    plt.suptitle(
        f"hi {high} lo {low} blur {blur} multi {boost_contrast}"  # . σ: {σ}.\nPré-traitements: {'Flou: %d' % blur if blur else 'Aucun'}"
        f"        /: {brightness_of(image)/current_edges_brightness:.3f}; Δ: {brightness_of(image) - current_edges_brightness:.3f}"
        # f"\nContraste: {contrast_of(image)}"
    )

    plt.savefig(
        batch_output_filename(
            image_path, low=low, high=high, blur=blur, boost_contrast=boost_contrast
        )
    )
    progress.done += 1
    print(image_path.stem, boost_contrast, blur, low, high, progress.done)
    # progress_bar.update(
    #     task,
    #     advance=1,
    #     description=Path(
    #         batch_output_filename(broken, low=low, high=high, blur=blur)
    #     ).name,
    # )


@synchronized
def do_batch(
    filepaths: list[Path],
    blur_values: list[int],
    high_values: list[int],
    contrast_boost_values: list[int],
    σ: int = 3,
):
    """
    Does a batch of edge detection on image image with all combinations of provided low & high tresholds.
    - image: A tuple of paths (broken bone image, baseline or "healed" image) or a single path (broken bone only)
    - use_blur: whether to preprocess the image with blur
    - σ: sets aperture size.
    """
    progress = Progress(
        done=0,
        total=len(filepaths)
        * len(high_values)
        * len(blur_values)
        * len(contrast_boost_values),
    )
    for filepath in filepaths:
        image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)

        # with Progress() as progress_bar:
        # task = progress_bar.add_task(
        #     "[blue]Processing...",
        #     total=len(list(high_values))
        #     * len(list(low_values) * len(list(blur_values))),
        # )

        for blur in blur_values:
            for boost_contrast in contrast_boost_values:
                for high in high_values:
                    # for low in low_values:
                    do_one(
                        filepath,
                        image,
                        int(high / 2),
                        high,
                        σ,
                        blur,
                        progress,
                        boost_contrast=boost_contrast,
                    )


if __name__ == "__main__":
    opts = docopt(__doc__)
    target = Path(opts["TARGET"])
    if Path(target).is_dir():
        filenames = list(target.glob("*"))
    else:
        filenames = [Path(target)]
    threshold_values = list(
        range(int(opts["START"]), int(opts["STOP"]), int(opts["STEP"]))
    )

    do_batch(
        # image=(target, target.parent / f"{target.with_suffix('').name}_baseline.png"),
        filepaths=filenames,
        high_values=threshold_values,
        blur_values=list(map(int, opts["--blur"])),
        contrast_boost_values=list(map(int, opts["--contrast"])),
        σ=int(opts["--aperture"]),
    )
