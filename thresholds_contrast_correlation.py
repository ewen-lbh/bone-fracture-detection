import json
from pathlib import Path
from typing import NamedTuple, Optional

import cv2
import matplotlib.pyplot as plt
from rich.progress import Progress, TaskID

from detect import brightness_of, contrast_of, detect_edges

TARGET_BRIGHTNESS = 15


class DataPoint(NamedTuple):
    filename: str
    brightness: float
    contrast: float
    lo: float
    hi: float

    def __getitem__(self, key):
        return getattr(self, key)


threshold_values = list(range(10, 200, 20))

cached: list[DataPoint] = []

if (Path(__file__).parent / "correlations.json").exists():
    cached = [
        DataPoint(**point)
        for point in json.loads(
            (Path(__file__).parent / "correlations.json").read_text()
        )
    ]


def cached_point(filename: str) -> Optional[DataPoint]:
    if filename in [point.filename for point in cached]:
        return [point for point in cached if point.filename == filename][0]


def performance(brightness_of_edges: float, original_brightness: float) -> float:
    return 1 / max(1, abs(brightness_of_edges - TARGET_BRIGHTNESS))


def process_file(bar: Progress, task_id: TaskID, filename: Path) -> DataPoint:
    if cached_point(filename.name):
        point = cached_point(filename.name)
        bar.update(task_id, advance=len(threshold_values) ** 2)
        return point

    best_threshold = (10, 10)
    image = cv2.imread(str(filename))
    best_threshold_brightness = brightness_of(detect_edges(image, *best_threshold)[1])
    for lo in threshold_values:
        for hi in threshold_values:
            _, edges = detect_edges(image, lo, hi)
            if performance(brightness_of(edges), brightness_of(image)) > performance(
                best_threshold_brightness, brightness_of(image)
            ):
                best_threshold = (lo, hi)
                best_threshold_brightness = brightness_of(edges)

            bar.update(
                task_id, advance=1, description=f"{filename.name} {lo:3d}:{hi:3d}"
            )

    return DataPoint(
        filename=str(filename),
        brightness=brightness_of(image),
        contrast=contrast_of(image),
        lo=best_threshold[0],
        hi=best_threshold[1],
    )


def process_files(filenames: list[Path]) -> list[DataPoint]:
    with Progress() as progress_bar:
        task_id = progress_bar.add_task(
            "Computing data points", total=len(threshold_values) ** 2 * len(filenames)
        )
        for filename in filenames:
            data.append(process_file(progress_bar, task_id, filename))
            (Path(__file__).parent / "correlations.json").write_text(
                json.dumps([point._asdict() for point in data])
            )

    return data


def attr(name: str) -> list:
    return [getattr(point, name) for point in data]


# ax = plt.axes(projection="3d")
# ax.plot3D(attr("contrast"), attr("bright"), attr("lo"))
# ax.set_ylabel("brightness")
# ax.set_xlabel("contrast")
# ax.set_zlabel("optimal low threshold")

data = process_files(list(Path("./datasets/radiopaedia").glob("*.png")))

fig, axes = plt.subplots(1, 2)

french = {
    "contrast": "du constraste",
    "brightness": "de la luminosité",
}

for current_axis, varying in enumerate(("contrast", "brightness")):
    data.sort(key=lambda o: o[varying])
    axes[current_axis].set_title(f"En fonction {french[varying]}")
    axes[current_axis].plot(attr(varying), attr("hi"), label="seuil haut", color="red")
    axes[current_axis].plot(
        attr(varying), attr("lo"), label="seuil bas", color="skyblue"
    )
    axes[current_axis].legend()

plt.show()
