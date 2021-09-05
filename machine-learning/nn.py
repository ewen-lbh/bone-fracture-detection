from pathlib import Path
from numpy import ndarray
from cv2 import imread, resize, IMREAD_GRAYSCALE
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Input, optimizers, callbacks
from tensorflow.data import Dataset
import keras.backend as K 

def build_model(width, height) -> Model:
    """Build a 3D convolutional neural network model."""

    inputs = Input((width, height, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    return Model(inputs, outputs, name="3dcnn")

def load_image(path: Path, as_size: int) -> ndarray:
    return resize(
        imread(path, IMREAD_GRAYSCALE),
        (as_size, as_size)
    )

def load_dataset(directory: Path) -> list[ndarray]:
	images = np.asarray([ load_image(file, 320) for file in directory.iterdir() ]).astype('float32')

    # Normalisation
	mean = np.mean(images)
	std = np.std(images)
	images = (images - mean) / std
	
	if K.image_data_format() == "channels_first":
		images = np.expand_dims(images, axis=1)		   #Extended dimension 1
	if K.image_data_format() == "channels_last":
		images = np.expand_dims(images, axis=3)             #Extended dimension 3(usebackend tensorflow:aixs=3; theano:axixs=1) 
	return images

def train(epochs: int, initial_learning_rate: float):
    model = build_model(width=320, height=320)
    train_dataset = 

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    # Define callbacks.
    checkpoint_cb = callbacks.ModelCheckpoint(
        "model.h5", save_best_only=True
    )
    early_stopping_cb = callbacks.EarlyStopping(monitor="val_acc", patience=15)

    # Train the model, doing validation at the end of each epoch
    epochs = 100
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )
