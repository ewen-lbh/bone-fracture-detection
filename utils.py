from typing import Iterable, Union


def mean(o: Iterable[Union[float, int]]) -> float:
    values = list(o)
    return sum(values) / len(values)

def flatten_2D(o: Iterable):
    flat = []
    for row in o:
        for item in row:
            flat.append(item)
    return flat
