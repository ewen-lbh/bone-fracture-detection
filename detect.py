from pathlib import Path
import json
from typing import Any, Optional, TypeVar

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress

from angles import display_lines, get_lines_probabilistic
from detect_tilt import image_tilt
from nptyping import NDArray
from utils import *


def vectorize(image):
    # convert cells of 3-valued lists (RGB) to a single value (greyscale)
    image = np.array([[0 if r == 0 else 1 for r in row] for row in image])
    return Bitmap(image).trace()


def is_broken(angles: list[float], ε: float = 10) -> bool:
    """
    If the maximum offset with a vertical angle is less than ε for all angles, the bone is not broken
    """
    tau = 2 * np.pi
    deg = lambda rad: rad / tau * 180
    print("[" + " ".join(f"{int(deg(angle))}°" for angle in angles) + "]")
    return max(map(deg, angles)) >= ε


def is_white(pixel: float) -> bool:
    return pixel > 0.75


def center_of(image: np.ndarray) -> np.ndarray:
    # return image[:, len(image[0])//4: -len(image[0])//4]
    return image


def contrast_of(image: np.ndarray) -> float:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std()


def boost_contrast(image: np.ndarray) -> np.ndarray:
    return 4 * image


def grayscale_of(image: NDArray[Any, Any, 3]) -> NDArray[Any, Any]:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]


def brightness_of(image: Union[NDArray[Any, Any, 3], NDArray[Any, Any]]) -> float:
    # RGB
    if len(image.shape) == 3:
        image = grayscale_of(image)
    return mean(flatten_2D(image))


def detect_edges(
    image: NDArray[Any, Any, 3], low: int, high: int, σ: int = 3, blur: int = 0, boost_contrast: bool = False
) -> tuple[NDArray[Any, Any, 3], NDArray[Any, Any]]:
    σ, low, high = map(int, (σ, low, high))
    contrast_was_boosted = False

    if contrast_of(image) < 20 and boost_contrast:
        image *= 4
        contrast_was_boosted = True
    # elif contrast_of(image) < 25:
    #     image *= 2
    #     contrast_was_boosted = True

    if contrast_was_boosted:
        print(f"boosted contrast is {contrast_of(image)}")

    if blur:
        image = cv2.blur(image, (blur, blur))
    edges = cv2.Canny(image, low, high, apertureSize=σ, L2gradient=True)
    return image, edges


def save_figure(image_path: Path, save: Optional[Path] = None):
    image = cv2.imread(str(image_path))
    print(f"contrast is {contrast_of(image)}")
    original, edges = detect_edges(image, low=40, high=120, blur=3)
    lines = list(get_lines_probabilistic(center_of(edges), gap=5, length=20))
    if not lines:
        print(f"error: no lines detected for {image}")
        return
    broken = is_broken([angle for _, _, angle in lines])
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle(
        f"Détecté comme {'cassé' if broken else 'sain'}\n"
        f"cont: {contrast_of(image)} lum: {brightness_of(image)}\n"
        # f"tilt: {image_tilt(lines)/(2*np.pi)*180}° #segments: {len(lines)}\n"
        f"outlum: {brightness_of(center_of(edges))} lumratio: {brightness_of(center_of(edges))/brightness_of(image)}"
    )
    ax[0].imshow(original)
    # ax[1].imshow(edges)
    display_lines(ax[1], center_of(edges), lines)

    if save:
        plt.savefig(str(save))
    else:
        plt.show()

    print(
        f'{image_path}: Detected as {"broken" if broken else "healthy"}',
        end="\n\n",
    )


if __name__ == "__main__":
    with Progress() as bar:
        files = list(Path("datasets/various").glob("*.png"))
        task = bar.add_task("[blue]Processing", total=len(files))
        for testfile in files:
            save_figure(testfile, save=Path("line-detection") / testfile.name)
            bar.advance(task)
