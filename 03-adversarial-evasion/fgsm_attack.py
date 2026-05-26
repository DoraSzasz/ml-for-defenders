"""Fast Gradient Sign Method evasion attack — for defenders.

Companion code for Post 3 of the ml-for-defenders series.

WHAT THIS DOES
    Loads the MNIST classifier produced by `train_classifier.py`, picks one
    test image, and uses the Fast Gradient Sign Method (FGSM) to add a
    small perturbation that flips the model's prediction. Displays the
    original image, the perturbation, and the adversarial image side by
    side, with both predictions printed in the titles.

    By default this runs the *iterative* variant of FGSM (Kurakin, 2017),
    which respects the same L-infinity budget EPSILON but splits it into
    STEPS small gradient steps and re-projects after each one. Single-step
    FGSM is a special case: set STEPS = 1.

    This is a textbook demonstration of MITRE ATLAS technique:
        AML.T0015 — Evade ML Model
        https://atlas.mitre.org/techniques/AML.T0015
    inside tactic:
        AML.TA0007 — Defense Evasion
        https://atlas.mitre.org/tactics/AML.TA0007

WHAT THIS DOES NOT DO
    Run against any model, dataset, or system that is not the local MNIST
    classifier you just trained. Do not point this at a third-party model,
    a cleared device, a clinical AI appliance, or anything covered by a
    license or contract you do not own. Adversarial-robustness testing in
    a regulated environment is a sanctioned activity with paperwork
    attached — not a Substack lab.

PARAMETERS WORTH TURNING
    EPSILON       — total L-infinity perturbation budget. Start at 0.15.
                    Lower it until the attack fails. That's your margin.
    STEPS         — number of iterative steps. 1 = classic single-step
                    FGSM. 10 = reliable iterative FGSM at the same budget.
    IMAGE_INDEX   — which MNIST test image to attack. Try 0, 4, 10, anything.

SETUP
    pip install -r requirements.txt
    python train_classifier.py     # produces mnist_model.h5

RUN
    python fgsm_attack.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

MODEL_PATH = "mnist_model.h5"
EPSILON = 0.40           # total L-infinity budget — turn this dial
STEPS = 10               # 1 = single-step FGSM, >1 = iterative FGSM
IMAGE_INDEX = 0          # MNIST test image 0 is a "7" — matches the article's example


def load_inputs() -> tuple[tf.Variable, int]:
    """Load MNIST test data and return one image as a tf.Variable + its label."""
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    image = (x_test[IMAGE_INDEX] / 255.0).astype(np.float32)
    image = tf.Variable(image.reshape(1, 28, 28, 1))
    label = int(y_test[IMAGE_INDEX])
    return image, label


def fgsm_attack(
    model: tf.keras.Model,
    image: tf.Variable,
    label: int,
    loss_fn: tf.keras.losses.Loss,
    epsilon: float,
    steps: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Iterative FGSM. Returns (adversarial_image, total_perturbation).

    Single-step FGSM is the special case steps == 1.
    """
    alpha = epsilon / steps
    original = tf.identity(image)
    adv = tf.Variable(image)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            prediction = model(adv)
            loss = loss_fn(tf.constant([label]), prediction)
        gradient = tape.gradient(loss, adv)

        # step in the direction that increases loss
        adv = adv + alpha * tf.sign(gradient)
        # project back into the L-infinity ball around the original image
        adv = tf.clip_by_value(adv, original - epsilon, original + epsilon)
        # keep pixel values valid
        adv = tf.clip_by_value(adv, 0.0, 1.0)
        adv = tf.Variable(adv)

    perturbation = adv - original
    return adv, perturbation


def predict(model: tf.keras.Model, image: tf.Tensor) -> int:
    """Return the model's argmax prediction for a single image."""
    return int(tf.argmax(model(image), axis=1).numpy()[0])


def show(original: tf.Tensor, perturbation: tf.Tensor, adversarial: tf.Tensor,
         pred_orig: int, pred_adv: int, epsilon: float) -> None:
    """Three-panel figure: original | perturbation | adversarial."""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"original\nprediction: {pred_orig}")
    plt.imshow(original[0, ..., 0], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"perturbation\n(epsilon = {epsilon})")
    plt.imshow(perturbation[0, ..., 0], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"adversarial\nprediction: {pred_adv}")
    plt.imshow(adversarial[0, ..., 0], cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    model = tf.keras.models.load_model(MODEL_PATH)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    image, label = load_inputs()
    adversarial, perturbation = fgsm_attack(
        model, image, label, loss_fn, EPSILON, STEPS
    )

    pred_orig = predict(model, image)
    pred_adv = predict(model, adversarial)

    print(f"True label:             {label}")
    print(f"Original prediction:    {pred_orig}")
    print(f"Adversarial prediction: {pred_adv}")
    print(f"Epsilon (L-inf budget): {EPSILON}")
    print(f"Steps:                  {STEPS}")

    if pred_orig != pred_adv:
        print("\n=> ATLAS AML.T0015 (Evade ML Model) demonstrated.")
    else:
        print("\n=> Attack did not flip the prediction at this budget. "
              "Try raising EPSILON, raising STEPS, or picking a different IMAGE_INDEX.")

    show(image, perturbation, adversarial, pred_orig, pred_adv, EPSILON)


if __name__ == "__main__":
    main()
