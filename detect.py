from angles import display_lines, get_lines, get_lines_probabilistic
from pathlib import Path
from svgpathtools import svg2paths
from potrace import Bitmap
import cv2
import potrace
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import Progress

def vectorize(image):
    # convert cells of 3-valued lists (RGB) to a single value (greyscale)
    image = np.array([ [ 0 if r == 0 else 1 for r in row ] for row in image ])
    return Bitmap(image).trace()

def retprint(a):
    print(a)
    return a

def is_broken(angles: list[float], ε: float = 20) -> bool:
    """
    If the maximum offset with a vertical angle (90° or 270°) is less than ε for all angles, the bone is not broken
    """
    print([ a/2*np.pi*180 for a in angles])
    return retprint(max(abs(a - (90 if 0 < a < 180 else 270)) for a in angles)) < ε

def is_white(pixel: float) -> bool:
    return pixel > 0.75

def center_of(image: np.ndarray) -> np.ndarray:
    # return image[:, len(image[0])//4: -len(image[0])//4]
    return image

def contrast_of(image: Path) -> float:
    return cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2GRAY).std()

def brightness_of(image: Path) -> float:
    return cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2HSV).value()

def detect_edges(filename: Path, low: int, high: int, σ: int = 3, blur: int = 0):
    σ, low, high = map(int, (σ, low, high))

    image = cv2.imread(str(filename))
    if blur:
        image = cv2.blur(image, (blur, blur))
    edges = cv2.Canny(image, low, high, apertureSize=σ, L2gradient=True)
    return image, edges


def save_figure(image: Path):
    print(f"contrast is {contrast_of(image)}")
    original, edges = detect_edges(image, low=40, high=120, blur=3)
    lines = list(get_lines_probabilistic(center_of(edges), gap=5, length=20))
    broken = is_broken([ angle for _, _, angle in lines ])
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle(f"Détecté comme {'cassé' if broken else 'sain'}")
    ax[0].imshow(original)
    display_lines(ax[1], center_of(edges), lines, save=Path("line-detection") / image.name)
    print(f"{str(image)}: Detected as {'broken' if broken else 'healthy'}")

if __name__ == "__main__":
    with Progress() as bar:
        files = list(Path("datasets/various").glob("*.png"))
        task = bar.add_task("[blue]Processing", total=len(files))
        for testfile in files:
            save_figure(testfile)
            bar.advance(task)
