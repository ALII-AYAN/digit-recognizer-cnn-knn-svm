#!/usr/bin/env python3
"""Download the MNIST IDX files into ``data/MNIST``.

Two sources are supported:

1. Kaggle (https://www.kaggle.com/datasets/hojjatk/mnist-dataset) — requires the
   ``kaggle`` package and an API token (``~/.kaggle/kaggle.json``).
2. The original mirror at https://yann.lecun.com/exdb/mnist/ — no login needed,
   files are gzipped and are byte-identical to the Kaggle ones.

Usage
-----
    python scripts/download_mnist.py                 # auto: Kaggle first, then mirror
    python scripts/download_mnist.py --source mirror
    python scripts/download_mnist.py --source kaggle --dest data/MNIST
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]

# Several public mirrors of the same four files, tried in order.
MIRROR_URLS = (
    "https://yann.lecun.com/exdb/mnist/{name}.gz",
    "https://storage.googleapis.com/cvdf-datasets/mnist/{name}.gz",
    "https://ossci-datasets.s3.amazonaws.com/mnist/{name}.gz",
)
KAGGLE_DATASET = "hojjatk/mnist-dataset"


def _fetch(url: str, dest_file: Path) -> None:
    """Stream a URL to disk. The mirrors reject requests without a User-Agent."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (digit-recognizer downloader)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response, open(dest_file, "wb") as out:
        shutil.copyfileobj(response, out)


def download_from_mirror(dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = dest / name
        if target.exists():
            print(f"  {name}: already present, skipping")
            continue

        gz_path = dest / f"{name}.gz"
        downloaded = False
        for template in MIRROR_URLS:
            url = template.format(name=name)
            print(f"  {name}: downloading {url}")
            try:
                _fetch(url, gz_path)
                downloaded = True
                break
            except Exception as exc:  # noqa: BLE001 - network errors vary
                print(f"    failed ({exc}), trying next mirror")
                gz_path.unlink(missing_ok=True)

        if not downloaded:
            return False

        with gzip.open(gz_path, "rb") as f_in, open(target, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        print(f"  {name}: saved ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def download_from_kaggle(dest: Path) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("  the 'kaggle' package is not installed (pip install kaggle)")
        return False

    try:
        api = KaggleApi()
        api.authenticate()
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {KAGGLE_DATASET} from Kaggle ...")
        api.dataset_download_files(KAGGLE_DATASET, path=str(dest), unzip=True, quiet=False)
    except Exception as exc:  # noqa: BLE001 - auth errors are common here
        print(f"  Kaggle download failed: {exc}")
        print("  hint: place your API token at ~/.kaggle/kaggle.json (chmod 600)")
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the MNIST dataset")
    parser.add_argument("--dest", type=Path, default=PROJECT_ROOT / "data" / "MNIST")
    parser.add_argument("--source", choices=("auto", "kaggle", "mirror"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Destination: {args.dest}")

    sources = ("kaggle", "mirror") if args.source == "auto" else (args.source,)
    for source in sources:
        print(f"Trying source: {source}")
        ok = download_from_kaggle(args.dest) if source == "kaggle" else download_from_mirror(args.dest)
        if ok and all((args.dest / name).exists() for name in FILES):
            print(f"\nDone. {len(FILES)} files available in {args.dest}")
            return
        print(f"  source '{source}' incomplete, trying the next one.\n")

    sys.exit("Could not download MNIST. See https://www.kaggle.com/datasets/hojjatk/mnist-dataset "
             "for manual download, then place the four IDX files in data/MNIST/.")


if __name__ == "__main__":
    main()
