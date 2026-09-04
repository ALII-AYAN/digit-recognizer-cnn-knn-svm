# Digit Recognizer — CNN vs KNN vs SVM

Handwritten digit recognition that trains and compares three classifiers on the same
data and the same train/test split:

| Model | Input | Notes |
| --- | --- | --- |
| **CNN** | raw 28×28 pixels | Small Conv–Pool–Conv–Pool–Dense network, softmax output |
| **KNN** (k=3) | HOG descriptors | Classical baseline, no training time |
| **SVM** (RBF) | HOG descriptors | Classical baseline, strongest of the two on HOG |

The project ships a **PyQt5 desktop app** that loads an image, runs all three models
and shows the CNN confidence distribution next to a majority-vote analysis.

```
data/MNIST (IDX)  ─┐
                   ├─> merge ─> split ─> [CNN branch]  ─> cnn_model.keras
data/custom/<0-9> ─┘                  ─> [HOG branch]  ─> knn_model.joblib
                                                       └> svm_model.joblib
```

## Features

- Clean, modular code: data loading, features, models, training, inference are separate modules.
- One single stratified split shared by all three models, so the accuracy comparison is fair.
- Fully configurable from the command line — **no hard-coded Windows paths**.
- CLI inference (`python -m src.predict`) and a GUI app (`python -m app.gui`).
- Optional image polarity inversion, for datasets with black digits on a white background.
- Metrics and the accuracy chart are written to `outputs/` (`metrics.json`, PNG).
- Headless-safe plotting and reproducible seeds.

## Project structure

```
digit-recognizer/
├── app/
│   └── gui.py              # PyQt5 desktop application
├── src/
│   ├── data.py             # MNIST IDX reader + custom folder loader
│   ├── features.py         # HOG feature extraction (shared train/inference params)
│   ├── models.py           # model factories (CNN, KNN, SVM)
│   ├── train.py            # training + evaluation + saving pipeline
│   ├── predict.py          # single-image CLI inference
│   └── utils.py            # seeding, plotting, filesystem helpers
├── scripts/
│   └── download_mnist.py    # fetch the Kaggle MNIST files (or public mirrors)
├── tests/
│   └── test_smoke.py       # fast unit tests (no training)
├── data/                   # datasets live here (git-ignored)
│   ├── MNIST/              # train-images-idx3-ubyte, train-labels-idx1-ubyte
│   └── custom/             # 0/ 1/ ... 9/ folders of images
├── models/                 # trained artefacts (git-ignored)
├── outputs/                # metrics.json + accuracy chart (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/digit-recognizer.git
cd digit-recognizer

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Requirements: Python 3.9–3.11. TensorFlow installs a CPU build by default, which is
enough for this network (15 epochs on MNIST takes a few minutes on a modern CPU).

## Data preparation

Download MNIST from Kaggle — <https://www.kaggle.com/datasets/hojjatk/mnist-dataset>
— and put **all four** IDX files in `data/MNIST/` (60k train + 10k official test):

```
data/MNIST/
├── train-images-idx3-ubyte   # 60,000 training images
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte    # 10,000 official test images
└── t10k-labels-idx1-ubyte
```

Or fetch them automatically:

```bash
python scripts/download_mnist.py            # Kaggle first, then public mirrors
python scripts/download_mnist.py --source mirror
```

Kaggle also ships the same files under dotted names (`train-images.idx3-ubyte`);
both spellings are detected, and `.gz` files are decompressed on read.

Put your own images in `data/custom/`, one folder per digit:

```
data/custom/
├── 0/  img001.png ...
├── 1/
...
└── 9/
```

> **Polarity matters.** MNIST stores white digits on a black background. If your custom
> images are black digits on a white background, train and predict with `--invert`
> (CLI) or tick the *Invert image* box in the GUI. Mixing polarities silently ruins accuracy.

## Usage

### 1. Train all three models

```bash
python -m src.train                                   # MNIST + data/custom
python -m src.train --no-custom                       # MNIST only
python -m src.train --invert-custom --max-per-class 500
python -m src.train --epochs 20 --batch-size 128 --n-neighbors 5 --svm-c 10
python -m src.train --limit-samples 6000              # quick smoke run
```

Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mnist-dir` | `data/MNIST` | Folder with the IDX files |
| `--custom-dir` | `data/custom` | Custom dataset root |
| `--max-per-class` | `500` | Cap custom images per digit (`0` = all) |
| `--invert-custom` | off | Invert custom image polarity |
| `--epochs` / `--batch-size` | `15` / `64` | CNN training |
| `--eval-split` | `auto` | `mnist-test` (official t10k), `random`, or `auto` |
| `--test-size` | `0.2` | Fraction held out for custom images / random mode |
| `--n-neighbors` / `--svm-c` | `3` / `1.0` | KNN *k*, SVM regularisation `C` |
| `--limit-samples` | `0` | Subsample training for a fast run (`0` = all) |
| `--seed` | `42` | Reproducibility |

**How the evaluation set is chosen** (`--eval-split`):

| Value | Behaviour |
| --- | --- |
| `mnist-test` | Train on the official 60k split, evaluate on the untouched 10k `t10k` split. Custom images are split 80/20: the larger part joins training, the smaller joins the test set. |
| `auto` (default) | Same as `mnist-test` when the `t10k-*` files exist, otherwise falls back to `random`. |
| `random` | Pool everything and take one stratified split. |

All three models always see the **exact same** test set, so the comparison is fair.

Outputs:

```
models/cnn_model.keras, knn_model.joblib, svm_model.joblib
outputs/metrics.json
outputs/model_accuracy_comparison.png
```

> **Runtime note.** The SVM is an RBF kernel fitted on ~50k HOG vectors, which is
> O(n²) in memory and time — expect it to take much longer than the CNN on a laptop.
> Use `--limit-samples` (e.g. `10000`) while iterating, then run the full training once.

### 2. Classify one image (CLI)

```bash
python -m src.predict --image path/to/digit.png
python -m src.predict --image path/to/digit.png --invert
```

```
Image      : path/to/digit.png
CNN        : 7  (confidence 99.41%)
KNN        : 7
SVM        : 7
Majority   : 7  -> All models agree.
```

### 3. Desktop app (GUI)

```bash
python -m app.gui
```

Click **Select Image**, and the app shows the 28×28 preview, the three predictions,
the agreement analysis and the CNN softmax bar chart.

### 4. Tests

```bash
pytest -q
```

## Results

Numbers below come from `outputs/metrics.json` after a run — replace them with your own.

| Model | Test accuracy |
| --- | --- |
| CNN | — |
| KNN | — |
| SVM | — |

Typical MNIST figures for this setup: **CNN ≈ 98–99%**, **SVM (RBF on HOG) ≈ 96–98%**,
**KNN (k=3 on HOG) ≈ 94–96%**. Your numbers will differ slightly once custom images
are mixed in, since those are usually harder than MNIST.

The accuracy chart is saved to `outputs/model_accuracy_comparison.png`; add it to
`assets/` and reference it here to show it on the repo page:

```markdown
![Model accuracy comparison](assets/model_accuracy_comparison.png)
```

## What was cleaned up for this version

Compared with the original two scripts, the following were removed or rewritten:

- **Hard-coded absolute paths** (`C:\Users\ayana\...`) → CLI arguments with defaults relative to the project root.
- **Duplicated logic** between the training and inference scripts → shared `src/predict.preprocess_image()` and `src/features.py`, so HOG parameters can never drift apart.
- **Unused imports** (`os`, `struct` and `QSizePolicy` leftovers) and emoji-heavy `print` spam → concise, level-tagged logging.
- **Brittle export paths** (plot written to a personal folder) → everything lands in `outputs/`, which is git-ignored.
- **Two independent `train_test_split` calls** that only *happened* to match → one split reused by all models.
- **Evaluated on re-split training data** → now evaluates on the official 10k `t10k` split when those files are present, which is the benchmark others publish.
- **Crash on small custom datasets**: `train_test_split(stratify=...)` raised `test_size = 4 should be >= 10 classes` when the custom folder had few images → automatic fallback to a shuffled split with a warning.
- **Image buffer bug in the GUI**: the NumPy array backing the `QImage` could be garbage-collected → the array is now kept as an instance attribute.
- **Model format**: `.h5` → native `.keras` (the loader still accepts an existing `.h5`).

## License

[MIT](LICENSE) — free to use and modify with attribution.
