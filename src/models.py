"""Model factories for the three classifiers compared in this project.

All three share the same interface (``fit`` / ``predict``) so ``src/train.py`` can
treat them uniformly.
"""

from __future__ import annotations

from typing import Dict

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def build_cnn(input_shape=(28, 28, 1), num_classes: int = 10):
    """A small CNN: Conv -> Pool -> Conv -> Pool -> Dense -> Softmax."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_knn(n_neighbors: int = 3) -> KNeighborsClassifier:
    """K-Nearest Neighbours classifier on HOG features."""
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def build_svm(c: float = 1.0, gamma: str = "scale") -> SVC:
    """RBF-kernel SVM classifier on HOG features."""
    return SVC(kernel="rbf", gamma=gamma, C=c)


def build_classical_models(n_neighbors: int = 3, c: float = 1.0) -> Dict[str, object]:
    """Return the KNN and SVM estimators keyed by name."""
    return {
        "KNN": build_knn(n_neighbors=n_neighbors),
        "SVM": build_svm(c=c),
    }
