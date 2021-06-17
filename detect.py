from pathlib import Path
from svgpathtools import svg2paths
from edge_detection_batch import detect_edges
from potrace import Bitmap
import cv2
import potrace
import numpy as np
import matplotlib.pyplot as plt

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


orig, edges = detect_edges(Path("datasets/various/coude_clair_2.png"), low=40, high=120)
center = edges[:, len(edges[0]) // 4: -len(edges[0]) // 4 ]
cv2.imwrite("datasets/various/coude_clair_2_edges.png", center)
# convert -resize 500% -filter gaussian -blur 0x4 -modulate 180 coude_clair_2.png coude_clair_2_edges.bmp
# potrace --svg coude_clair_2_edges.bmp
paths, _ = svg2paths("datasets/various/coude_clair_2_edges.svg")
print(paths)
