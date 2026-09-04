"""HOG feature extraction used by the classical KNN / SVM models.

The parameters below must stay identical between training (``src/train.py``) and
inference (``src/predict.py``, ``app/gui.py``), otherwise the models receive
features from a different distribution.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)


def extract_hog_feature(image: np.ndarray) -> np.ndarray:
    """HOG descriptor of a single 28x28 grayscale image -> 1-D array."""
    return hog(image, **HOG_PARAMS)


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """HOG descriptors for a batch of images -> array of shape (N, n_features)."""
    return np.asarray([extract_hog_feature(img) for img in images], dtype=np.float64)
