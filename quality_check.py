"""
This modules provides functions to check if the photo taken is:
- sharp enough (not too blurry)
- centered enough
(- has good incidence angles of X-rays) 
"""

from pathlib import Path
from numpy import ndarray
from cv2 import Laplacian, cvtColor, COLOR_RGB2GRAY, CV_64F, imread

def good_enough(image: ndarray) -> bool:
    return not blurry(image)

def blurry(image: ndarray) -> bool:
    variance_Δ = Laplacian(cvtColor(image, COLOR_RGB2GRAY), CV_64F).var()
    print(f"variance(Δ(image)) = {variance_Δ}")

    return variance_Δ



if __name__ == "__main__":
    print(min(blurry(imread(str(path))) for path in (Path(__file__).parent / "datasets" / "encyclopaedia").iterdir()))
    # min: 2.4 
    # blurry: 2.4

    # for image_file in (Path(__file__).parent / "datasets" / "encyclopaedia").iterdir():
    #     if "blurry" in image_file.name: continue
    #     image = imread(str(image_file))
    #     print(image_file)
    #     is_blurry = blurry(image)
    #     print(f"{is_blurry=}")
