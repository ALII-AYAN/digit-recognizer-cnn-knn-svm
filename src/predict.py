"""Command-line inference: classify a single digit image with CNN, KNN and SVM.

Usage
-----
    python -m src.predict --image path/to/digit.png
    python -m src.predict --image path/to/digit.png --invert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

from src.data import IMAGE_SIZE
from src.features import extract_hog_feature

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def preprocess_image(
    image_path: str | Path, invert: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """Read an image and prepare it for both model branches.

    Returns ``(cnn_input, hog_feature, resized)`` where ``cnn_input`` has shape
    ``(1, 28, 28, 1)`` (float32 in [0, 1]) and ``hog_feature`` has shape
    ``(1, n_features)``. Returns ``(None, None, None)`` if the file is unreadable.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    if invert:
        img = cv2.bitwise_not(img)

    resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    cnn_input = (resized.astype("float32") / 255.0)[np.newaxis, ..., np.newaxis]
    hog_feature = extract_hog_feature(resized).reshape(1, -1)
    return cnn_input, hog_feature, resized


def load_models(models_dir: Path = MODELS_DIR) -> dict[str, object]:
    """Load the three trained models from disk."""
    import tensorflow as tf

    cnn_path = models_dir / "cnn_model.keras"
    if not cnn_path.exists():  # backwards compatibility with the old .h5 export
        cnn_path = models_dir / "cnn_model.h5"
    if not cnn_path.exists():
        sys.exit(f"No CNN model found in {models_dir}. Run 'python -m src.train' first.")

    return {
        "CNN": tf.keras.models.load_model(cnn_path),
        "KNN": joblib.load(models_dir / "knn_model.joblib"),
        "SVM": joblib.load(models_dir / "svm_model.joblib"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a digit image with CNN / KNN / SVM")
    parser.add_argument("--image", type=Path, required=True, help="Path to the digit image")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--invert", action="store_true",
                        help="Invert the image (black digits on white background)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cnn_input, hog_feature, _ = preprocess_image(args.image, invert=args.invert)
    if cnn_input is None:
        sys.exit(f"Could not read image: {args.image}")

    models = load_models(args.models_dir)

    cnn_probs = models["CNN"].predict(cnn_input, verbose=0)[0]
    cnn_pred = int(np.argmax(cnn_probs))
    knn_pred = int(models["KNN"].predict(hog_feature)[0])
    svm_pred = int(models["SVM"].predict(hog_feature)[0])

    votes = [cnn_pred, knn_pred, svm_pred]
    majority = max(set(votes), key=votes.count)
    agreement = votes.count(majority)
    verdict = {3: "All models agree.", 2: "Two models agree.", 1: "Models disagree."}[agreement]

    print(f"Image      : {args.image}")
    print(f"CNN        : {cnn_pred}  (confidence {cnn_probs[cnn_pred] * 100:.2f}%)")
    print(f"KNN        : {knn_pred}")
    print(f"SVM        : {svm_pred}")
    print(f"Majority   : {majority}  -> {verdict}")

    print("\nCNN confidence per class")
    for digit, prob in enumerate(cnn_probs):
        print(f"  {digit}: {prob * 100:5.2f}%")


if __name__ == "__main__":
    main()
