import cv2
import numpy as np
from pathlib import Path
import sys

img = cv2.imread(sys.argv[1])
blurred = cv2.blur(img, (3, 3))
edges = cv2.Canny(blurred, 50, 200)

points = np.argwhere(edges > 0)
y_1, x_1 = points.min(axis = 0)
y_2, x_2 = points.max(axis = 0)

cropped = img[y_1:y_2, x_1:x_2]
(Path(sys.argv[1]).parent / "autocropped").mkdir(exist_ok=True)
cv2.imwrite(str(Path(sys.argv[1]).parent / "autocropped" / Path(sys.argv[1]).name), cropped)
