"""
On entraîne un réseau dans le but de lui faire diminuer l'écart à un intervalle de contrastes/luminosités acceptables
correspondant à des choix de seuils convenables.
On utilisera un CNN, qui est souvent utilisé lorsque les données traitées sont des images, qui entraîne des vecteurs de grandes dimensions.
Ici, la fonction de coût n'est pas simplement l'écart de certitude au résultat attendu dans le set de données d'entraînement,
car la réponse donnée par le réseau n'a pas de relation simple au critère d'évaluation de son choix.
On utilise donc une fonction de coût qui compare le contraste et la luminosité idéale à celle atteinte en utilisant les seuils proposés par le réseau.
Une réponse est considérée comme juste (ou acceptable) si cet écart est inférieur à un seuil de tolérance fixé ε.
"""
from pathlib import Path
from math import sqrt
from typing import *
import numpy as np
from PIL import Image
from nptyping import NDArray

from detect import brightness_of, detect_edges


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
        h, w = image.shape

        for i, j in zip(range(h - 2), range(w - 2)):
            yield image[i : i + 3, j : j + 3], i, j

    def forward(self, input: NDArray):
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for region, i, j in self.regions(input):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))

        return output


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
                output[i, j] = np.amax(region, axis=(0, 1))

        return output


class Softmax(Layer):
    weights: NDArray
    biases: NDArray
    max_value: float

    def __init__(self, nodes, max_value: float = 1) -> None:
        self.nodes = nodes
        self.biases = np.zeros(nodes)
        self.max_value = max_value

    def weights(self, input) -> NDArray:
        w, h, num_filters = input.shape
        input_length = w * h * num_filters
        return np.random.randn(input_length, self.nodes) / input_length

    def forward(self, input):
        totals = np.dot(input.flatten(), self.weights(input)) + self.biases
        exponentiateds = np.exp(totals)

        return (exponentiateds / np.sum(exponentiateds, axis=0)) * self.max_value


class Network:
    """
    Add new layers by "calling" the class again.
    Set loss or is_valid at any call to set them.
    Last call setting loss or is_valid will be used.
    """

    layers: list[Layer]
    loss: Callable[[NDArray, NDArray], float]
    is_valid: Callable[[float], bool]

    def __init__(
        self, layers: list[Layer], loss=lambda x: 0, is_valid=lambda loss: False
    ) -> None:
        self.layers = layers
        self.loss = loss
        self.is_valid = is_valid

    def forward(self, input):
        output = input
        for layer in self.layers:
            print(f"Doing layer {layer}")
            output = layer.forward(output)
            print(f"Output shape: {output.shape}")
        loss = self.loss(output, input)
        return input, loss, self.is_valid(loss)

    __call__ = forward


ε = 5


def Features(**args) -> dict:
    base = {
        "luminosity": 0,
    }

    for key, value in args.items():
        if key in base:
            base[key] = value
        else:
            raise KeyError(f"{key} is not a valid Features key")

    return base


def loss(output: NDArray, input: NDArray) -> float:
    got = Features()
    want = Features(luminosity=15)

    high_treshold, low_treshold = output.tolist()[:2]
    print(input.shape)
    edges, _ = detect_edges(image=input, low=low_treshold, high=high_treshold)

    got["luminosity"] = brightness_of(edges)

    deviations = [abs(got[feature] - want[feature]) for feature in Features._fields]
    return sqrt(sum(map(lambda x: x**2, deviations)))


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

for i, image_path in enumerate(Path("datasets/radiopaedia").glob("*.png")):
    image = np.array(Image.open(image_path))
    _, loss, accuracy = network(image)

    print(f"{image.stem}: {loss:.2f}, {accuracy}")


# https://victorzhou.com/blog/intro-to-cnns-part-1/
