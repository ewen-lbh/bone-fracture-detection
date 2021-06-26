from os import access
from pathlib import Path
import matplotlib.pyplot as plt
import yaml


data = yaml.load(Path("optimal-values.yaml").read_text())
data_as_list = [ {"name": k, **v} for k,v in data.items() ]
data_as_list.sort(key=lambda o: o["contrast"])

attr = lambda name: list(map((lambda o: o[name]), data_as_list))

plt.plot(attr("contrast"), attr("lo"))
plt.plot(attr("contrast"), attr("hi"))

plt.show()
