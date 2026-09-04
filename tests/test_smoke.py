"""Fast tests that do not require training or a TensorFlow install."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import pytest

from src.data import IMAGE_SIZE, Dataset, combine, load_custom_dataset, load_mnist,\
    load_mnist_splits
from src.features import HOG_PARAMS, extract_hog_feature, extract_hog_features

skimage = pytest.importorskip("skimage")


def _rand_images(n: int = 5) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n, *IMAGE_SIZE), dtype=np.uint8)


def test_hog_feature_shape_is_stable():
    feature = extract_hog_feature(_rand_images(1)[0])
    # 28x28, 8x8 cells -> 3x3 cells, 2x2 blocks -> 2x2 blocks, 9 orientations each
    assert feature.shape == (2 * 2 * 2 * 2 * HOG_PARAMS["orientations"],)


def test_hog_batch_matches_single():
    images = _rand_images(4)
    batch = extract_hog_features(images)
    assert batch.shape[0] == 4
    assert np.allclose(batch[2], extract_hog_feature(images[2]))


def test_classical_models_fit_and_predict():
    from src.models import build_knn, build_svm

    images = _rand_images(30)
    labels = np.array([i % 10 for i in range(30)], dtype=np.uint8)
    X = extract_hog_features(images)

    for model in (build_knn(n_neighbors=3), build_svm(c=1.0)):
        model.fit(X, labels)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)
        assert set(preds) <= set(range(10))


def test_predict_preprocess_shapes():
    import cv2

    from src.predict import preprocess_image

    path = Path(__file__).parent / "_tmp_digit.png"
    cv2.imwrite(str(path), _rand_images(1)[0])
    try:
        cnn_input, hog_feature, resized = preprocess_image(path)
        assert cnn_input.shape == (1, *IMAGE_SIZE, 1)
        assert 0.0 <= cnn_input.min() and cnn_input.max() <= 1.0
        assert hog_feature.shape[0] == 1
        assert resized.shape == IMAGE_SIZE
    finally:
        path.unlink(missing_ok=True)


def test_custom_dataset_loader(tmp_path: Path):
    import cv2

    for digit in (0, 1):
        folder = tmp_path / str(digit)
        folder.mkdir()
        for i in range(3):
            cv2.imwrite(str(folder / f"{i}.png"), _rand_images(1)[0])

    data = load_custom_dataset(tmp_path, max_per_class=2)
    assert len(data) == 4
    assert set(np.unique(data.labels)) == {0, 1}
    assert data.images.shape[1:] == IMAGE_SIZE
    assert data.cnn_input.shape == (4, *IMAGE_SIZE, 1)
    assert data.cnn_input.max() <= 1.0


def _write_idx(tmp_path: Path, prefix: str, images: np.ndarray, labels: np.ndarray) -> None:
    """Write an IDX train/test pair using the given prefix ('train' or 't10k')."""
    (tmp_path / f"{prefix}-images-idx3-ubyte").write_bytes(
        struct.pack(">IIII", 2051, len(images), *IMAGE_SIZE) + images.tobytes()
    )
    (tmp_path / f"{prefix}-labels-idx1-ubyte").write_bytes(
        struct.pack(">II", 2049, len(labels)) + labels.tobytes()
    )


def test_mnist_idx_reader(tmp_path: Path):
    images = _rand_images(6)
    labels = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
    _write_idx(tmp_path, "train", images, labels)

    data = load_mnist(tmp_path)
    assert np.array_equal(data.images, images)
    assert np.array_equal(data.labels, labels)


def test_mnist_splits_with_t10k(tmp_path: Path):
    """Kaggle's download contains 4 files -> the official test split must be picked up."""
    train_images, train_labels = _rand_images(20), np.arange(20, dtype=np.uint8) % 10
    test_images, test_labels = _rand_images(8), np.arange(8, dtype=np.uint8) % 10
    _write_idx(tmp_path, "train", train_images, train_labels)
    _write_idx(tmp_path, "t10k", test_images, test_labels)

    splits = load_mnist_splits(tmp_path)
    assert len(splits.train) == 20
    assert splits.test is not None and len(splits.test) == 8
    assert np.array_equal(splits.test.labels, test_labels)


def test_mnist_splits_without_t10k(tmp_path: Path):
    _write_idx(tmp_path, "train", _rand_images(10), np.arange(10, dtype=np.uint8) % 10)
    splits = load_mnist_splits(tmp_path)
    assert len(splits.train) == 10
    assert splits.test is None  # falls back to a random split in train.py


def test_kaggle_dotted_filenames(tmp_path: Path):
    """The Kaggle mirror also ships 'train-images.idx3-ubyte' style names."""
    images, labels = _rand_images(4), np.array([1, 2, 3, 4], dtype=np.uint8)
    (tmp_path / "train-images.idx3-ubyte").write_bytes(
        struct.pack(">IIII", 2051, 4, *IMAGE_SIZE) + images.tobytes()
    )
    (tmp_path / "train-labels.idx1-ubyte").write_bytes(
        struct.pack(">II", 2049, 4) + labels.tobytes()
    )
    data = load_mnist(tmp_path)
    assert len(data) == 4
    assert np.array_equal(data.labels, labels)


def test_gzipped_idx_files(tmp_path: Path):
    import gzip

    images, labels = _rand_images(5), np.array([5, 6, 7, 8, 9], dtype=np.uint8)
    (tmp_path / "train-images-idx3-ubyte.gz").write_bytes(
        gzip.compress(struct.pack(">IIII", 2051, 5, *IMAGE_SIZE) + images.tobytes())
    )
    (tmp_path / "train-labels-idx1-ubyte.gz").write_bytes(
        gzip.compress(struct.pack(">II", 2049, 5) + labels.tobytes())
    )
    data = load_mnist(tmp_path)
    assert np.array_equal(data.images, images)
    assert np.array_equal(data.labels, labels)


def test_train_split_strategy_uses_official_test():
    """build_splits must train on 60k and evaluate on the untouched 10k t10k split."""
    from src.data import DatasetSplits
    from src.train import build_splits

    args = argparse.Namespace(
        eval_split="auto", test_size=0.2, seed=42,
        limit_samples=0, mnist_dir="ignored", no_custom=True, max_per_class=0,
    )
    mnist_train = Dataset(_rand_images(40), np.arange(40, dtype=np.uint8) % 10)
    mnist_test = Dataset(_rand_images(10), np.arange(10, dtype=np.uint8) % 10)
    custom = Dataset(_rand_images(20), np.arange(20, dtype=np.uint8) % 10)

    train_data, test_data, mode = build_splits(DatasetSplits(mnist_train, mnist_test), custom, args)
    assert "official" in mode
    assert len(train_data) == 40 + 16      # full MNIST train + 80% of custom
    assert len(test_data) == 10 + 4        # full t10k test + 20% of custom

    # No custom image may appear in both sets -> build_splits guards against leakage.
    assert len(train_data) + len(test_data) == 40 + 10 + 20


def test_train_split_strategy_random_fallback():
    from src.data import DatasetSplits
    from src.train import build_splits

    args = argparse.Namespace(
        eval_split="random", test_size=0.25, seed=42,
        limit_samples=0, mnist_dir="ignored", no_custom=True, max_per_class=0,
    )
    mnist_train = Dataset(_rand_images(40), np.arange(40, dtype=np.uint8) % 10)
    custom = Dataset(_rand_images(20), np.arange(20, dtype=np.uint8) % 10)

    train_data, test_data, mode = build_splits(DatasetSplits(mnist_train, None), custom, args)
    assert mode == "random stratified split"
    assert len(train_data) == 45 and len(test_data) == 15


def test_combine_validates_and_concatenates():
    a = Dataset(_rand_images(3), np.array([1, 2, 3], dtype=np.uint8))
    b = Dataset(_rand_images(2), np.array([4, 5], dtype=np.uint8))
    merged = combine([a, b])
    assert len(merged) == 5

    with pytest.raises(ValueError):
        Dataset(_rand_images(3), np.array([1, 2], dtype=np.uint8))
