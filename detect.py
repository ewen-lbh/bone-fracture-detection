from angle_research_2 import display_lines, get_lines
from pathlib import Path
from svgpathtools import svg2paths
from edge_detection_batch import detect_edges
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

def is_broken(edges: list[list[float]]) -> bool:
    center = edges[len(edges[0]) // 6: -len(edges[0]) // 6 ]
    for curve in vectorize(center):
        for segment in curve.segments:
            pass


def is_white(pixel: float) -> bool:
    return pixel > 0.75

def center_of(image: np.ndarray) -> np.ndarray:
    # return image[:, len(image[0])//4: -len(image[0])//4]
    return image

def save_figure(image: Path):
    _, edges = detect_edges(image, low=40, high=120)
    lines = get_lines(center_of(edges))
    display_lines(center_of(edges), lines, save=Path("line-detection") / image.name)

with Progress() as bar:
    files = list(Path("datasets/various").glob("*.png"))
    task = bar.add_task("[blue]Processing", total=len(files))
    for testfile in files:
        save_figure(testfile)
        bar.advance(task)
# cv2.imwrite("datasets/various/coude_clair_2_edges.png", center)
# # convert -resize 500% -filter gaussian -blur 0x4 -modulate 180 coude_clair_2.png coude_clair_2_edges.bmp
# # potrace --svg coude_clair_2_edges.bmp
# paths, _ = svg2paths("datasets/various/coude_clair_2_edges.svg")
# print(paths)
