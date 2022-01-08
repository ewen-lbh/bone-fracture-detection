from pathlib import Path

from tensorflow.keras.models import Model
from tensorflow.keras import layers, Input, optimizers, callbacks
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data import Dataset

training_dataset: Dataset = image_dataset_from_directory(
    "datasets/encyclopaedia", labels=None
)

"""
Reinforcement learning
======================

* Actions:
- 1: Increase lower threshold by some ε 
- 2: Decrease lower threshold by some ε  
- 3: Increase upper threshold by some ε  
- 4: Decrease upper threshold by some ε  

* Environment:
"""


dense = layers.Dense(units=16)
