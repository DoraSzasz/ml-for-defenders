"""Train a tiny MNIST digit classifier — companion code for post 3.

Companion code for Post 3 of the ml-for-defenders series.

WHAT THIS DOES
    Trains a small convolutional neural network on the MNIST handwritten
    digit dataset for three epochs and saves the trained model to
    `mnist_model.h5`. The saved model is the target for the adversarial
    evasion demo in `fgsm_attack.py`.

WHAT THIS DOES NOT DO
    Touch any clinical data, any PHI, any real medical-imaging model, or
    any system you do not control. MNIST is a public dataset of handwritten
    digits. The point of this script is to give you a local, disposable
    target so the adversarial-evasion lesson lands without putting anyone
    at risk.

SETUP
    pip install -r requirements.txt

RUN
    python train_classifier.py
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_PATH = "mnist_model.h5"
EPOCHS = 3
BATCH_SIZE = 128


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST, add a channel axis, and normalize pixel values to [0, 1]."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train[..., np.newaxis] / 255.0).astype(np.float32)
    x_test = (x_test[..., np.newaxis] / 255.0).astype(np.float32)
    return x_train, y_train, x_test, y_test


def build_model() -> tf.keras.Model:
    """A small CNN — deliberately simple, easy to attack, easy to read."""
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10),  # logits, not softmax — required by the attack
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    x_train, y_train, x_test, y_test = load_mnist()
    model = build_model()
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
    )
    model.save(MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
