"""Train and compare CNN, KNN and SVM on MNIST + a custom digit dataset.

Usage
-----
    python -m src.train --mnist-dir data/MNIST --custom-dir data/custom

Run ``python -m src.train --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.data import Dataset, combine, load_custom_dataset, load_mnist_splits
from src.features import extract_hog_features
from src.models import build_classical_models, build_cnn
from src.utils import ensure_dir, plot_accuracy_comparison, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN / KNN / SVM digit classifiers")
    parser.add_argument("--mnist-dir", type=Path, default=PROJECT_ROOT / "data" / "MNIST",
                        help="Folder holding the MNIST IDX files")
    parser.add_argument("--custom-dir", type=Path, default=PROJECT_ROOT / "data" / "custom",
                        help="Custom dataset root, laid out as <digit>/<image>")
    parser.add_argument("--no-custom", action="store_true", help="Train on MNIST only")
    parser.add_argument("--max-per-class", type=int, default=500,
                        help="Cap number of custom images per digit (0 = all)")
    parser.add_argument("--invert-custom", action="store_true",
                        help="Invert custom images (use for black digits on white background)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--eval-split",
        choices=("auto", "mnist-test", "random"),
        default="auto",
        help="Evaluation set: 'mnist-test' uses the official t10k split (10k images), "
             "'random' re-splits everything; 'auto' prefers t10k when the files exist",
    )
    parser.add_argument("--n-neighbors", type=int, default=3, help="KNN neighbours")
    parser.add_argument("--svm-c", type=float, default=1.0, help="SVM regularisation C")
    parser.add_argument("--limit-samples", type=int, default=0,
                        help="Subsample the merged dataset (0 = use everything). "
                             "Handy for a quick smoke run: SVM is O(n^2).")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_indices(
    n: int, labels: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/test indices, with a graceful fallback for tiny datasets.

    Stratification needs at least one sample per class on each side, which a small
    custom folder (e.g. 20 images over 10 digits) cannot satisfy. In that case we
    fall back to a plain shuffled split instead of crashing.
    """
    stratify = labels
    if stratify is not None and len(np.unique(stratify)) > n * min(test_size, 1 - test_size):
        print("      [warn] too few samples to stratify the custom split, shuffling instead")
        stratify = None

    return train_test_split(
        np.arange(n), test_size=test_size, random_state=seed, stratify=stratify
    )


def build_splits(
    splits, custom: Dataset | None, args: argparse.Namespace
) -> tuple[Dataset, Dataset, str]:
    """Decide which images are used for training and which for evaluation.

    ``auto`` / ``mnist-test``: train on the official 60k MNIST train split and
    evaluate on the official 10k t10k split — the standard, comparable benchmark.
    Custom images are still split (``--test-size``) so their held-out part joins the
    test set and the training part joins training, with no leakage between the two.

    ``random`` (or when the t10k files are missing): pool everything and take one
    stratified split. Every model sees the exact same test set either way.
    """
    use_official = args.eval_split == "mnist-test" or (
        args.eval_split == "auto" and splits.test is not None
    )

    if use_official and splits.test is not None:
        custom_train = custom_test = None
        if custom is not None:
            c_idx = split_indices(len(custom), custom.labels, args.test_size, args.seed)
            custom_train, custom_test = custom.subset(c_idx[0]), custom.subset(c_idx[1])

        train_data = combine([splits.train, custom_train])
        test_data = combine([splits.test, custom_test])
        return train_data, test_data, "official MNIST test split (t10k)"

    pooled = combine([splits.train, splits.test, custom])
    idx = split_indices(len(pooled), pooled.labels, args.test_size, args.seed)
    return pooled.subset(idx[0]), pooled.subset(idx[1]), "random stratified split"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    models_dir = ensure_dir(args.models_dir)
    outputs_dir = ensure_dir(args.outputs_dir)

    # ------------------------------------------------------------------ data
    print("[1/5] Loading MNIST ...")
    splits = load_mnist_splits(args.mnist_dir)
    print(f"      MNIST train: {len(splits.train)}")

    custom = None
    if not args.no_custom and args.custom_dir.exists():
        print("[1/5] Loading custom dataset ...")
        custom = load_custom_dataset(
            args.custom_dir,
            max_per_class=args.max_per_class or None,
            invert=args.invert_custom,
        )
        print(f"      custom images: {len(custom)}")

    train_data, test_data, eval_mode = build_splits(splits, custom, args)
    print(f"      evaluation mode: {eval_mode}")

    # Subsample only the training set — the test set stays intact so accuracy stays comparable.
    if args.limit_samples and args.limit_samples < len(train_data):
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(train_data), args.limit_samples, replace=False)
        train_data = train_data.subset(keep)
        # Also cap the test set for the O(n^2) SVM, otherwise it dominates runtime.
        if len(test_data) > 2000:
            test_data = test_data.subset(rng.choice(len(test_data), 2000, replace=False))
        print(f"      subsampled to  : {len(train_data)} train / {len(test_data)} test")

    print(f"      train samples  : {len(train_data)}")
    print(f"      test samples   : {len(test_data)}")

    results: dict[str, float] = {}

    # ------------------------------------------------------------------- CNN
    print("[2/5] Training CNN ...")
    X_train_cnn, X_test_cnn = train_data.cnn_input, test_data.cnn_input
    y_train, y_test = train_data.labels, test_data.labels

    cnn = build_cnn()
    cnn.summary()
    cnn.fit(
        X_train_cnn,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        verbose=2,
    )
    cnn_loss, cnn_acc = cnn.evaluate(X_test_cnn, y_test, verbose=0)
    results["CNN"] = float(cnn_acc)
    print(f"      CNN test accuracy: {cnn_acc * 100:.2f}% (loss {cnn_loss:.4f})")

    # --------------------------------------------------------- HOG + KNN/SVM
    print("[3/5] Extracting HOG features ...")
    X_train_hog = extract_hog_features(train_data.images)
    X_test_hog = extract_hog_features(test_data.images)

    print("[4/5] Training KNN and SVM ...")
    fitted: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, model in build_classical_models(args.n_neighbors, args.svm_c).items():
        model.fit(X_train_hog, y_train)
        fitted[name] = model
        predictions[name] = model.predict(X_test_hog)
        results[name] = float(accuracy_score(y_test, predictions[name]))
        print(f"      {name} test accuracy: {results[name] * 100:.2f}%")

    # --------------------------------------------------------------- persist
    print("[5/5] Saving models, metrics and plot ...")
    cnn.save(models_dir / "cnn_model.keras")
    joblib.dump(fitted["KNN"], models_dir / "knn_model.joblib")
    joblib.dump(fitted["SVM"], models_dir / "svm_model.joblib")

    metrics = {
        "accuracies": results,
        "cnn_loss": float(cnn_loss),
        "eval_mode": eval_mode,
        "n_train": int(len(train_data)),
        "n_test": int(len(test_data)),
        "epochs": args.epochs,
        "report": classification_report(y_test, predictions["SVM"], output_dict=True, zero_division=0),
    }
    (outputs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_accuracy_comparison(results.keys(), [v * 100 for v in results.values()],
                             outputs_dir / "model_accuracy_comparison.png")

    print("\nSummary")
    for name, acc in results.items():
        print(f"  {name:<4} {acc * 100:6.2f}%")
    print(f"\nModels  -> {models_dir}")
    print(f"Outputs -> {outputs_dir}")


if __name__ == "__main__":
    main()
