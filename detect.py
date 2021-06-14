from edge_detection_batch import detect_edges
from potrace import Bitmap
import numpy as np

def vectorize(image):
    return Bitmap(image)

def is_broken(edges: list[list[float]]) -> bool:
    center = edges[len(edges[0]) // 4: -len(edges[0]) // 4 ]
    for row in center:
        for pixel in row:
            

def is_white(pixel: float) -> bool:
    return pixel > 0.75
