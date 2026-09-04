# Datasets

This folder is **git-ignored** — never commit raw datasets.

Get MNIST from the Kaggle mirror you are using:
<https://www.kaggle.com/datasets/hojjatk/mnist-dataset>

That download contains **four** IDX files (unzip them all into `data/MNIST/`):

```
data/MNIST/
├── train-images-idx3-ubyte   # 60,000 training images
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte    # 10,000 official test images
└── t10k-labels-idx1-ubyte
```

Kaggle also ships the same files under dotted names (`train-images.idx3-ubyte`).
Both spellings are detected automatically, and `.gz` files are decompressed on read.

Or let the script fetch them for you:

```bash
python scripts/download_mnist.py            # tries Kaggle, then public mirrors
python scripts/download_mnist.py --source mirror
```

## Why the t10k files matter

If `t10k-*` is present, training uses the official 60k train split and evaluation
uses the untouched 10k test split — the standard benchmark that your numbers can be
compared against. If they are missing, everything is pooled and split randomly
(`--test-size 0.2`), which inflates accuracy slightly.

## Custom images

```
data/custom/
├── 0/
├── 1/
...
└── 9/
```

Read as grayscale, resized to 28×28, capped at **500 per digit** (`--max-per-class`).
MNIST stores white digits on black; if yours are dark digits on a light background,
use `--invert-custom` when training and `--invert` when predicting.
