from math import sqrt
from typing import Iterable, Union, TypeVar


def mean(o: Iterable[Union[float, int]]) -> float:
    values = list(o)
    return sum(values) / len(values)


def flatten_2D(o: Iterable):
    flat = []
    for row in o:
        for item in row:
            flat.append(item)
    return flat


# needed to remove pylint(cell-var-from-loop), see https://stackoverflow.com/a/67928238/9943464
def access(o, key):
    return o.get(key)


def norm(vector):
    return sqrt(sum(map(lambda x: x ** 2, vector)))


def checkmark(o):
    return "[bold green]✔[/]" if o else "[bold red]✘[/]"

T = TypeVar("T")
def partition(o: Iterable[T], layout: Union[list[int], tuple[int]]) -> list[list[T]]:
    """
    Chunks o into chunks of sizes given by layout.

    >>> partition([1, 2, 3, 4, 5, 6, 7, 8, 9], (3, 3, 3))
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    >>> partition([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (4, 4, 4))
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    >>> partition([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (6, 2, 4))
    [[1, 2, 3, 4, 5, 6], [7, 8], [9, 10, 11, 12]]
    """
    # can't use [[]] * len(layout) as all sublists will be views of the same sublist
    partitions = []
    for _ in layout:
        partitions.append([])

    index_in_partition = 0
    current_partition = 0

    for item in o:
        partitions[current_partition].append(item)
        if index_in_partition == layout[current_partition] - 1:
            current_partition += 1
            index_in_partition = 0
        else:
            index_in_partition += 1
    
    return partitions
