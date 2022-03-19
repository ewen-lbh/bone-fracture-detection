"""
On entraîne un réseau dans le but de lui faire diminuer l'écart à un intervalle de contrastes/luminosités acceptables
correspondant à des choix de seuils convenables.
On utilisera un CNN, qui est souvent utilisé lorsque les données traitées sont des images, qui entraîne des vecteurs de grandes dimensions.
Ici, la fonction de coût n'est pas simplement l'écart de certitude au résultat attendu dans le set de données d'entraînement,
car la réponse donnée par le réseau n'a pas de relation simple au critère d'évaluation de son choix.
On utilise donc une fonction de coût qui compare le contraste et la luminosité idéale à celle atteinte en utilisant les seuils proposés par le réseau.
Une réponse est considérée comme juste (ou acceptable) si cet écart est inférieur à un seuil de tolérance fixé ε.
"""
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import *

import cv2
import numpy as np
from nptyping import NDArray
from numpy import ndarray
from PIL import Image
from rich import print
from rich.live import Live
from rich.table import Table

from detect import brightness_of, detect_edges
from utils import checkmark, flatten_2D, norm

ε = 5
CURRENT_IMAGE_STEM = ""


@dataclass
class LayerCache:
    shape: Tuple[int, ...] = ()
    flat: NDArray[Any, Any] = ndarray((1, 1))
    totals: NDArray[Any, Any] = ndarray((1, 1))


class Layer:
    """Abstract base class for layers, used for typing"""

    def forward(self, data: NDArray) -> NDArray:
        pass


class Convolution(Layer):
    num_filters: int
    filters: NDArray

    def __init__(self, num_filters: int):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def regions(self, image: NDArray):
        h, w = image.shape[:2]

        for i, j in zip(range(h - 2), range(w - 2)):
            yield image[i : i + 3, j : j + 3], i, j

    def forward(self, input: NDArray):
        h, w = input.shape[:2]
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for region, i, j in self.regions(input):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))

        return output

    def __str__(self):
        return f"Convolution({self.num_filters})"


class Pooling(Layer):
    def regions(self, image: NDArray):
        h, w, _ = image.shape

        for i in range(w // 2):
            for j in range(h // 2):
                yield image[(i * 2) : (i * 2 + 2), (j * 2) : (j * 2 + 2)], i, j

    def forward(self, input: NDArray):
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for region, i, j in self.regions(input):
            # Maximize over pixels (first 2 dimensions), not the filter number
            if 0 not in region.shape:
                try:
                    output[i, j] = np.amax(region, axis=(0, 1))
                except IndexError as e:
                    # print(f"warning: {e}")
                    pass
                except Exception as e:
                    print(
                        f"Tried pooling on region of shape {region.shape} into [{i}, {j}]"
                    )
                    raise e

        return output

    def __str__(self):
        return f"Pooling()"


class Softmax(Layer):
    biases: NDArray
    max_value: float
    cache: LayerCache

    def __init__(self, nodes, max_value: float = 1) -> None:
        self.nodes = nodes
        self.biases = np.zeros(nodes)
        self.max_value = max_value
        self.cache = LayerCache()

    def weights(self, input) -> NDArray:
        w, h, num_filters = input.shape
        input_length = w * h * num_filters
        return np.random.randn(input_length, self.nodes) / input_length

    def forward(self, input):
        self.cache.shape = input.shape
        flat = input.flatten()
        self.cache.flat = flat
        totals = np.dot(flat, self.weights(input)) + self.biases
        self.cache.totals = totals
        exponentiateds = np.exp(totals)

        return (exponentiateds / np.sum(exponentiateds, axis=0)) * self.max_value

    def backpropagate(self, loss_gradient):
        for i, gradient in enumerate(loss_gradient):
            if gradient == 0:
                continue

    def __str__(self):
        return f"Softmax({self.nodes})"


class Network:
    """
    Add new layers by "calling" the class again.
    Set loss or is_valid at any call to set them.
    Last call setting loss or is_valid will be used.
    """

    layers: list[Layer]
    loss: Callable[[NDArray, NDArray], float]
    is_valid: Callable[[float], bool]

    def __init__(self, layers: list[Layer], loss, is_valid) -> None:
        self.layers = layers
        self.loss = loss
        self.is_valid = is_valid

    def forward(self, input):
        output = input
        for layer in self.layers:
            # print(f"Doing layer {layer}")
            # print(f"Shapes: {output.shape} ->", end=" ")
            output = layer.forward(output)
            # print(f"{output.shape}")
        loss = self.loss(output, input)
        return input, loss, self.is_valid(loss)

    __call__ = forward


class Features(NamedTuple):
    luminosity: float = 0

    def __getitem__(self, key):
        return getattr(self, key)


def loss(output: NDArray[2], input: NDArray[Any, Any]) -> float:
    want = Features(luminosity=15)

    high_treshold, low_treshold = output.tolist()[:2]
    _, edges = detect_edges(
        image=cv2.cvtColor(input, cv2.COLOR_GRAY2RGB),
        low=low_treshold,
        high=high_treshold,
    )
    cv2.imwrite(
        str(
            Path(__file__).parent
            / "thresholds_by_ml_output"
            / (CURRENT_IMAGE_STEM + ".png")
        ),
        edges,
    )

    got = Features(luminosity=brightness_of(edges))

    deviations = [abs(got[feature] - want[feature]) for feature in Features._fields]
    return norm(deviations)


lessthan = lambda ε: lambda x: x <= ε

network = Network(
    loss=loss,
    is_valid=lessthan(ε),
    layers=[
        Convolution(8),
        Pooling(),
        Softmax(2),
    ],
)

results = Table(title="Results")
results.add_column("Image name")
results.add_column("Loss", justify="right")
results.add_column("Accurate?", justify="right")

with Live(results):
    for i, image_path in enumerate(Path("datasets/radiopaedia").glob("*.png")):
        CURRENT_IMAGE_STEM = image_path.stem
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        _, loss, accurate = network(image)

        results.add_row(
            image_path.stem.replace("-", " "), f"{loss:.2f}", checkmark(accurate)
        )


# https://victorzhou.com/blog/intro-to-cnns-part-1/
