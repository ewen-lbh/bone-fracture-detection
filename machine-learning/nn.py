from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Input, optimizers, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data import Dataset

training_dataset = image_dataset_from_directory(
    "datasets/MURA-v1.1-sorted",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(180, 180),
    batch_size=32,
)

validation_dataset = image_dataset_from_directory(
    "datasets/MURA-v1.1-sorted",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(180, 180),
    batch_size=32,
)

# TODO continue with tutorial https://keras.io/examples/vision/image_classification_from_scratch/
