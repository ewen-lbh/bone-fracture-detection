from os import access
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

data = yaml.load(Path("optimal-values.yaml").read_text())
# exclude blur!=6, we'll treat them separately
data_as_list = [{"name": k, **v} for k, v in data.items() if v["blur"] == 3]

attr = lambda name: list(map((lambda o: o.get(name, 0)), data_as_list))

# ax = plt.axes(projection="3d")
# ax.plot3D(attr("contrast"), attr("bright"), attr("lo"))
# ax.set_ylabel("brightness")
# ax.set_xlabel("contrast")
# ax.set_zlabel("optimal low threshold")

fig, axes = plt.subplots(1, 2)

french = {
    "contrast": "du constraste",
    "bright": "de la luminosité",
}

for current_axis, varying in enumerate(("contrast", "bright")):
    data_as_list.sort(key=lambda o: o[varying])
    axes[current_axis].set_title(f"En fonction {french[varying]}")
    axes[current_axis].plot(attr(varying), attr("hi"), label="seuil haut", color="red")
    axes[current_axis].plot(
        attr(varying), attr("lo"), label="seuil bas", color="skyblue"
    )
    axes[current_axis].legend()

plt.show()
