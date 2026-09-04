"""Dataset loading utilities.

Three data sources are supported and can be merged:

1. The official MNIST **train** split (60,000 images) from the IDX binary files
   (``train-images-idx3-ubyte`` / ``train-labels-idx1-ubyte``).
2. The official MNIST **test** split (10,000 images) from ``t10k-*`` files, used as
   the held-out benchmark instead of re-splitting the training set.
3. A custom folder of images organised as ``root/<label>/*.png``.

The Kaggle mirror (https://www.kaggle.com/datasets/hojjatk/mnist-dataset) ships the
same four IDX files in two naming styles (``train-images-idx3-ubyte`` and
``train-images.idx3-ubyte``); both are resolved automatically, and ``.gz``
compressed files are transparently decompressed.

All images are returned as uint8 grayscale arrays of shape ``(N, 28, 28)`` so that
the CNN branch (normalised to [0, 1]) and the HOG branch (raw intensities) stay
consistent with each other and with inference time.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

IMAGE_SIZE: Tuple[int, int] = (28, 28)

# Canonical file names, each with the accepted variants.
MNIST_FILES = {
    "train_images": ("train-images-idx3-ubyte", "train-images.idx3-ubyte"),
    "train_labels": ("train-labels-idx1-ubyte", "train-labels.idx1-ubyte"),
    "test_images": ("t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"),
    "test_labels": ("t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"),
}

# Backwards-compatible aliases used by older versions of this project.
MNIST_TRAIN_IMAGES = MNIST_FILES["train_images"][0]
MNIST_TRAIN_LABELS = MNIST_FILES["train_labels"][0]


@dataclass
class Dataset:
    """Container for images and labels."""

    images: np.ndarray  # uint8, shape (N, 28, 28)
    labels: np.ndarray  # uint8, shape (N,)

    def __len__(self) -> int:
        return len(self.labels)

    def __post_init__(self) -> None:
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length")

    @property
    def cnn_input(self) -> np.ndarray:
        """Normalised float32 images shaped ``(N, 28, 28, 1)`` for the CNN."""
        images = self.images.astype("float32") / 255.0
        return np.expand_dims(images, axis=-1)

    def subset(self, indices: np.ndarray) -> "Dataset":
        """Return a new Dataset containing only the given row indices."""
        return Dataset(self.images[indices], self.labels[indices])


@dataclass
class DatasetSplits:
    """Official MNIST splits. ``test`` is None when the t10k files are absent."""

    train: Dataset
    test: Optional[Dataset] = None


def _resolve(data_dir: Path, key: str) -> Optional[Path]:
    """Find an MNIST file by trying every accepted name, with or without .gz."""
    for name in MNIST_FILES[key]:
        for candidate in (data_dir / name, data_dir / f"{name}.gz"):
            if candidate.exists():
                return candidate
    return None


def _open_binary(path: Path):
    """Open a plain or gzipped IDX file in binary mode."""
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def load_mnist_images(filename: str | Path) -> np.ndarray:
    """Load an IDX3 image file -> uint8 array of shape (N, rows, cols)."""
    filename = Path(filename)
    with _open_binary(filename) as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{filename} is not a valid IDX3 image file (magic={magic})")
        images = np.frombuffer(f.read(), dtype=np.uint8)
    if images.size != num * rows * cols:
        raise ValueError(f"{filename}: expected {num * rows * cols} bytes, got {images.size}")
    return images.reshape(num, rows, cols)


def load_mnist_labels(filename: str | Path) -> np.ndarray:
    """Load an IDX1 label file -> uint8 array of shape (N,)."""
    filename = Path(filename)
    with _open_binary(filename) as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{filename} is not a valid IDX1 label file (magic={magic})")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if len(labels) != num:
        raise ValueError(f"{filename}: expected {num} labels, found {len(labels)}")
    return labels


def load_mnist(data_dir: str | Path) -> Dataset:
    """Load only the MNIST **training** split (60,000 images)."""
    data_dir = Path(data_dir)
    images_path = _resolve(data_dir, "train_images")
    labels_path = _resolve(data_dir, "train_labels")
    if images_path is None or labels_path is None:
        raise FileNotFoundError(
            f"MNIST training files not found in {data_dir}. Expected one of "
            f"{MNIST_FILES['train_images']} (optionally .gz)."
        )
    return Dataset(load_mnist_images(images_path), load_mnist_labels(labels_path))


def load_mnist_splits(data_dir: str | Path) -> DatasetSplits:
    """Load the MNIST train split and, when available, the official t10k test split."""
    data_dir = Path(data_dir)
    train = load_mnist(data_dir)

    test_images_path = _resolve(data_dir, "test_images")
    test_labels_path = _resolve(data_dir, "test_labels")
    if test_images_path is None or test_labels_path is None:
        return DatasetSplits(train=train, test=None)

    test = Dataset(
        load_mnist_images(test_images_path),
        load_mnist_labels(test_labels_path),
    )
    return DatasetSplits(train=train, test=test)


def load_custom_dataset(
    data_dir: str | Path,
    max_per_class: int | None = 500,
    invert: bool = False,
) -> Dataset:
    """Load a custom dataset laid out as ``data_dir/<digit>/<image files>``.

    Args:
        data_dir: root folder containing one sub-folder per digit (``0``..``9``).
        max_per_class: keep at most this many images per digit (``None`` = all).
        invert: set to True when your images are black-digits-on-white, which is
            the opposite polarity of MNIST (white digits on black background).
    """
    data_dir = Path(data_dir)
    images: List[np.ndarray] = []
    labels: List[int] = []

    for label in range(10):
        folder = data_dir / str(label)
        if not folder.is_dir():
            continue

        files = sorted(p for p in folder.iterdir() if p.is_file())
        if max_per_class is not None:
            files = files[:max_per_class]

        for path in files:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[data] unreadable image, skipped: {path}")
                continue
            if invert:
                img = cv2.bitwise_not(img)
            images.append(cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA))
            labels.append(label)

    if not images:
        raise FileNotFoundError(f"no images found under {data_dir}")

    return Dataset(np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.uint8))


def combine(datasets: List[Dataset]) -> Dataset:
    """Concatenate several datasets into one."""
    datasets = [d for d in datasets if d is not None and len(d) > 0]
    return Dataset(
        np.concatenate([d.images for d in datasets], axis=0),
        np.concatenate([d.labels for d in datasets], axis=0),
    )
