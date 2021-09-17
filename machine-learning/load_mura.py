from pathlib import Path
import numpy as np
import re
from string import ascii_lowercase
from rich import print

PATHS_PATTERN = re.compile(
    r"MURA-v1\.1/.+/XR_(?P<bodypart>[A-Z]+)/patient(?P<patient>\d+)/study(?P<study>\d+)_(?:positive|negative)/"
)


def arrange_dataset_files(name: str, input_directory: Path, output_directory: Path):
    """
    Re-arrange the dataset file structure, so that it contains only two subfolders:
    - 1/ for broken
    - 0/ for healthy.
    """

    (healthy := output_directory / "0").mkdir(parents=True, exist_ok=True)
    (broken := output_directory / "1").mkdir(exist_ok=True)

    for batch, label in read_csv(input_directory / f"{name}_labeled_studies.csv"):
        if match := PATHS_PATTERN.match(batch):
            metadata = match.groupdict()
        else:
            continue

        for image in (input_directory.parent / batch).iterdir():
            metadata["image"] = image.name.strip(
                "." + ascii_lowercase
            )  # keep only numbers
            print(label, *metadata.values())

            try:
                (
                    Path(healthy if label == "0" else broken)
                    / "{bodypart}-{patient}-{study}-{image}.png".format(**metadata)
                ).symlink_to(image)
            except FileExistsError:
                continue


def read_csv(file: Path):
    for line in file.read_text().splitlines():
        yield line.split(",")


if __name__ == "__main__":
    datasets_dir = Path(__file__).parent.parent / "datasets"
    for name in ("train", "valid"):
        arrange_dataset_files(
            name,
            datasets_dir / "MURA-v1.1",
            datasets_dir / "MURA-v1.1-sorted" / name,
        )
