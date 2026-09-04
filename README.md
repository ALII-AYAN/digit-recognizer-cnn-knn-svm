# Digit Recognizer — CNN vs KNN vs SVM

I built a handwritten digit classifier three different ways and compared them on the
same data and the same test split: a **convolutional neural network** trained on raw
28×28 pixels, and a **KNN** and an **RBF SVM** trained on **HOG descriptors**. On a
10,600-image test set the CNN reached **99.21%**, the SVM **98.14%** and the KNN
**97.36%**.

The interesting part was not the ranking — it was *where* each model broke. All
three did fine on MNIST and all three got noticeably worse on my own scanned
digits, but the CNN lost 3.75 points while the KNN lost 12.04. That gap is the
whole reason this project exists, and it is what the results section is about.

```
MNIST train (60,000) ─┐
                      ├─> merge ─> split ─> [raw pixels ] ─> CNN  ─> cnn_model.keras
custom (3,000) ───────┘                  ─> [HOG, 144-d] ─> KNN  ─> knn_model.joblib
                                                        └> SVM  ─> svm_model.joblib
```

Everything runs from one command, and there is a **PyQt5 desktop app** that loads
any digit image, runs all three models and shows the CNN's confidence distribution
next to a majority-vote analysis.

---

## Why three models

A CNN is the obvious answer for digit recognition in 2026, and it won. But "the CNN
won" is a much less useful result than "the CNN won by this much, and here is the
regime where it wins by more". I wanted a baseline that would make the comparison
informative rather than decorative, so I picked the two classical models that
actually work on this problem:

- **HOG + KNN** — no training at all, pure nearest-neighbour matching on gradient
  orientation histograms. It is the honest "how much do we need a neural network
  here?" baseline.
- **HOG + SVM (RBF)** — the strongest thing you can do on this dataset without
  deep learning. Before CNNs took over, HOG+SVM was the standard recipe.

Both use the **same 144-dimensional HOG descriptor**, so the difference between
them isolates the classifier, not the features. The CNN gets raw pixels, which is
the whole point: it is the only model here that learns its own features.

---

## Features

- **One shared split.** All three models train and are evaluated on exactly the same
  images, so the accuracy table is a real comparison rather than three numbers from
  three different experiments.
- **Official MNIST test split.** When the `t10k-*` files are present, training uses
  the 60k train split and evaluation uses the untouched 10k test split — the same
  benchmark everyone else publishes on.
- **Modular code.** Data loading, feature extraction, model definitions, training
  and inference are separate modules with no circular dependencies.
- **Three interfaces**: training CLI, single-image CLI, and a PyQt5 GUI.
- **No hard-coded paths.** Every directory, hyperparameter and seed is a CLI flag
  with a sensible default.
- **Explainability in the GUI.** Alongside the prediction you get the CNN's softmax
  distribution and an agreement analysis across all three models.
- **Reproducible.** One seed controls Python, NumPy and TensorFlow. Charts render
  headless, so training works over SSH.
- **13 tests** covering the IDX reader, the HOG pipeline and the split logic — they
  run in about 5 seconds without downloading anything.

---

## Project structure

```
digit-recognizer/
├── app/
│   └── gui.py                # PyQt5 desktop application
├── src/
│   ├── data.py               # MNIST IDX reader + custom folder loader
│   ├── features.py           # HOG extraction (shared by train and inference)
│   ├── models.py             # CNN / KNN / SVM factories
│   ├── train.py              # training, evaluation, metrics, charts
│   ├── predict.py            # single-image CLI + shared preprocessing
│   └── utils.py              # seeding, plotting, filesystem helpers
├── scripts/
│   └── download_mnist.py     # fetch the Kaggle MNIST files or a public mirror
├── tests/
│   └── test_smoke.py         # 13 tests
├── assets/                   # charts shown in this README
├── data/
│   ├── MNIST/                # the four IDX files (git-ignored)
│   └── custom/               # 0/ 1/ ... 9/ folders of images (git-ignored)
├── models/                   # trained artefacts (git-ignored)
├── outputs/                  # metrics.json + charts (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/aliayan/digit-recognizer.git
cd digit-recognizer

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.9–3.11. TensorFlow installs a CPU build by default, which is plenty: the
CNN has 225,034 parameters and 15 epochs over 62,400 images took me 6 min 20 s on a
laptop CPU. If you have a CUDA GPU, swap in `tensorflow>=2.13` and it drops to well
under a minute.

For the GUI you also need PyQt5, which is already in `requirements.txt`. On a
headless Linux box:

```bash
sudo apt-get install libxcb-cursor0    # Qt 5.15 needs this to find its platform plugin
```

---

## Data

Download MNIST from Kaggle — <https://www.kaggle.com/datasets/hojjatk/mnist-dataset>
— and put **all four** IDX files in `data/MNIST/`:

```
data/MNIST/
├── train-images-idx3-ubyte   # 60,000 training images
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte    # 10,000 official test images
└── t10k-labels-idx1-ubyte
```

Or let the script fetch them:

```bash
python scripts/download_mnist.py               # Kaggle first, then public mirrors
python scripts/download_mnist.py --source mirror
```

Kaggle ships the same files under dotted names (`train-images.idx3-ubyte`) too;
both spellings are detected, and `.gz` files are decompressed on read.

For my own images I collected 3,000 samples, 300 per digit, and laid them out as
`data/custom/0/`, `data/custom/1/`, ... They are read as grayscale and resized to
28×28.

> **Polarity matters.** MNIST stores white digits on a black background. My scans
> are the opposite, so I train with `--invert-custom` and predict with `--invert`.
> Mixing polarities does not crash — it just quietly destroys accuracy, which is
> much worse.

---

## Usage

### 1. Train all three models

```bash
python -m src.train --invert-custom
```

```
[1/5] Loading MNIST ...
      MNIST train: 60000
      custom images: 3000
      evaluation mode: official MNIST test split (t10k)
      train samples  : 62400
      test samples   : 10600
[2/5] Training CNN ...
      CNN test accuracy: 99.21% (loss 0.0261)
[3/5] Extracting HOG features ...
[4/5] Training KNN and SVM ...
      KNN test accuracy: 97.36%
      SVM test accuracy: 98.14%
[5/5] Saving models, metrics and plot ...

Summary
  CNN   99.21%
  SVM   98.14%
  KNN   97.36%
```

Every knob is a flag:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mnist-dir` | `data/MNIST` | Folder with the IDX files |
| `--custom-dir` | `data/custom` | Custom dataset root |
| `--max-per-class` | `500` | Cap custom images per digit (`0` = all) |
| `--invert-custom` | off | Invert custom image polarity |
| `--epochs` / `--batch-size` | `15` / `64` | CNN training |
| `--eval-split` | `auto` | `mnist-test`, `random`, or `auto` |
| `--test-size` | `0.2` | Fraction of custom images held out |
| `--n-neighbors` | `3` | KNN *k* |
| `--svm-c` | `1.0` | SVM regularisation `C` |
| `--limit-samples` | `0` | Subsample for a fast run (`0` = all) |
| `--seed` | `42` | Reproducibility |

A few I actually used:

```bash
python -m src.train --no-custom                    # MNIST only, no custom data
python -m src.train --epochs 20 --svm-c 5 --n-neighbors 5
python -m src.train --limit-samples 6000           # ~40 s smoke run
```

`--limit-samples` caps the test set at 2,000 as well, because the RBF SVM is O(n²)
and would otherwise dominate the runtime even on a "quick" run.

### 2. Classify one image

```bash
python -m src.predict --image scan_0042.png --invert
```

```
Image      : scan_0042.png
CNN        : 7  (confidence 99.41%)
KNN        : 7
SVM        : 7
Majority   : 7  -> All models agree.

CNN confidence per class
  0:  0.02%      5:  0.01%
  1:  0.04%      6:  0.03%
  2:  0.11%      7: 99.41%
  3:  0.07%      8:  0.09%
  4:  0.18%      9:  0.04%
```

The agreement analysis is more useful than it looks. When all three models agree
the answer is almost always right; when they disagree, the image is usually
genuinely ambiguous and the tie-break is worth a human glance.

### 3. Desktop app

```bash
python -m app.gui
```

Pick an image and the window shows the 28×28 preview, the three predictions, the
agreement verdict and a bar chart of the CNN's softmax output. The **Invert image**
checkbox handles dark digits on a light background without restarting.

```
+---------------------------------------------------------------+
|                    [ Select Image ]                           |
|              [x] Invert image (dark on light)                 |
|                   +--------------+                            |
|                   |  28x28       |                            |
|                   |  preview     |                            |
|                   +--------------+                            |
|   CNN predicted digit: 7 with confidence 99.41%   |  [bars]   |
|   KNN predicted digit: 7                          |   99.4%   |
|   SVM predicted digit: 7                          |   ...     |
|   Model agreement: All models agree on the digit. |           |
+---------------------------------------------------------------+
```

### 4. Tests

```bash
pytest -q
```

13 tests: IDX magic-number validation, the Kaggle dotted-filename variants, gzip
support, custom-folder loading, HOG shape stability, preprocessing shapes, and both
split strategies including the no-leakage guarantee that train + test equals the
total.

---

## Results

Trained on 62,400 images (60,000 MNIST train + 2,400 of my own), evaluated on
10,600 (10,000 MNIST `t10k` + 600 held-out custom). 15 epochs, batch size 64,
HOG 9 orientations / 8×8 cells / 2×2 blocks, `k=3`, `C=1.0`.

![Model accuracy comparison](assets/model_accuracy_comparison.png)

| Model | Features | Test accuracy | Errors | Train time | Inference |
| --- | --- | --- | --- | --- | --- |
| **CNN** | raw 28×28 pixels | **99.21%** | 84 / 10,600 | 6 min 20 s | 11 ms |
| SVM (RBF) | HOG, 144-d | 98.14% | 197 / 10,600 | 11 min 04 s | 2.4 ms |
| KNN (k=3) | HOG, 144-d | 97.36% | 280 / 10,600 | 0.8 s | 84 ms |

KNN has no training step — its 0.8 s is just building the search index — but it pays
for that at inference: every prediction is a nearest-neighbour scan over 62,400
stored vectors. The SVM is the opposite, slow to fit and fast to apply.

### Where the CNN still fails

![CNN confusion matrix](assets/confusion_matrix_cnn.png)

84 errors out of 10,600. They are not spread evenly — they concentrate on the digit
pairs that look alike:

| True → predicted | Count |
| --- | --- |
| 4 → 9 | 10 |
| 9 → 4 | 9 |
| 3 → 5, 5 → 3 | 7 each |
| 7 → 9 | 6 |
| 8 → 3 | 5 |

Every one of these is a case where a stroke is written slightly short or slightly
closed. A 4 whose diagonal does not quite meet the vertical becomes a 9; a 9 whose
loop does not close becomes a 4. That is the same ambiguity a human resolves from
context, so I do not expect it to go away without a bigger model or better data.

![Per-digit metrics](assets/per_class_metrics.png)

Macro F1 is 99.21. `0` is the easiest digit (F1 99.86) — it is the only one with no
straight strokes and no near neighbour — and `9` is the hardest (F1 98.49), dragged
down almost entirely by the 4/9 confusion.

![Training curves](assets/training_curves.png)

Validation accuracy plateaus around epoch 12 and validation loss starts creeping up
after epoch 15, which is why I stopped there.

### The finding I did not expect

Splitting the accuracy by source shows something the headline numbers hide:

| Model | MNIST `t10k` (10,000) | My scans (600) | Drop |
| --- | --- | --- | --- |
| CNN | 99.42% | 95.67% | **−3.75** |
| SVM | 98.57% | 91.00% | −7.57 |
| KNN | 98.04% | 86.00% | **−12.04** |

On MNIST all three models are within 1.4 points of each other — on that dataset,
arguably, you do not need a neural network. On images that were not produced by the
same pipeline as the training data, the ordering is the same but the gaps widen
dramatically. **The CNN loses less than a third as much accuracy as the KNN when the
input distribution shifts.**

That is what I take away from this: HOG is a hand-designed feature that happens to
suit MNIST's specific rendering — consistent stroke width, centred digits, uniform
background. Once my scans introduce different pen widths, off-centre digits and
slightly uneven background, the hand-designed descriptor stops describing the
important variation, and a model that learns its own features adapts. The KNN is
hit hardest because it has no learned abstraction to fall back on at all: it stores
training examples and measures raw distance, so any shift in the input moves every
distance at once.

![Sample predictions](assets/sample_predictions.png)

### Things I swept

I varied one setting at a time on the same splits to check my defaults were the
right ones.

**KNN, choice of *k*:**

| k | 1 | **3** | 5 | 7 |
| --- | --- | --- | --- | --- |
| Accuracy | 96.72% | **97.36%** | 97.29% | 96.98% |

k=1 overfits to individual noisy training samples; k=7 starts blurring class
boundaries. k=3 is the peak.

**SVM, regularisation *C*:**

| C | 1.0 | 5.0 | 10.0 |
| --- | --- | --- | --- |
| Accuracy | 98.14% | 98.16% | 98.11% |

Flat — a 0.05-point spread is noise on 10,600 samples. I kept the default `C=1.0`
because there was no reason not to, and because lower `C` fits faster.

**CNN, number of epochs:**

| Epochs | 5 | 10 | **15** | 20 |
| --- | --- | --- | --- | --- |
| Accuracy | 98.74% | 99.06% | **99.21%** | 99.23% |

Epoch 20 buys 0.02 points and starts overfitting, so 15 is the right stopping point.

**Features for the classical models** — this one mattered most:

| Model | Raw 784 pixels | HOG (144-d) |
| --- | --- | --- |
| KNN | 96.28% | **97.36%** |
| SVM (RBF) | 93.85% | **98.14%** |

HOG is worth 1.1 points to the KNN and **4.3 points** to the SVM. Raw pixels are a
terrible input for an RBF kernel: with 784 mostly-background dimensions, Euclidean
distance is dominated by where the digit happens to sit in the frame rather than by
its shape. HOG throws that positional sensitivity away and keeps the gradient
structure, which is exactly why it was the standard feature for this kind of work
before CNNs.

---

## Design notes

**Why the models share one split.** My first version called `train_test_split`
twice, once for the CNN branch and once for the HOG branch, with the same seed so
they *happened* to produce the same division. That is one changed default away from
silently comparing models on different test sets. Now the split is computed once
and handed to both branches.

**Why preprocessing lives in `src/predict.py`.** The GUI and the CLI both call
`preprocess_image()`. If the GUI had its own copy, the HOG parameters could drift
apart between training and inference, and accuracy would collapse with no error
message. Sharing the function makes that impossible.

**Why HOG parameters are module-level constants.** `HOG_PARAMS` in `src/features.py`
is imported by both the trainer and the predictor. There is exactly one place to
change them, and no way for the two to disagree.

**Why `.keras` instead of `.h5`.** Keras 3 deprecates the HDF5 format. The loader
still accepts an existing `.h5` file, so old checkpoints keep working.

**Why `errors="replace"` is not used here but is elsewhere.** The IDX format is
binary and length-prefixed, so a truncated file is detected by an explicit size
check rather than silently producing garbage arrays.

---

## Limitations

- **MNIST is not handwriting in the wild.** It is centred, size-normalised,
  deskewed, single-digit and cleanly binarised. Expect noticeably lower accuracy on
  real photographs of handwriting.
- **One digit per image.** There is no segmentation step, so multi-digit input
  produces whatever the network finds most salient rather than a sequence.
- **The SVM does not scale.** RBF fitting is O(n²); at 62,400 training samples it
  took 11 minutes and several gigabytes. `--limit-samples` exists for a reason.
- **Polarity is a manual setting.** There is no automatic detection of whether
  digits are light-on-dark or dark-on-light. Get it wrong and accuracy silently
  falls apart.
- **Custom data is small.** 3,000 images over 10 digits is enough to show the
  distribution-shift effect but not enough to fine-tune on.

---

## Dataset

MNIST via the Kaggle mirror <https://www.kaggle.com/datasets/hojjatk/mnist-dataset>,
which redistributes the original files by Yann LeCun, Corinna Cortes and
Christopher Burges (<http://yann.lecun.com/exdb/mnist/>).

```bibtex
@article{lecun1998gradient,
  title   = {Gradient-based learning applied to document recognition},
  author  = {LeCun, Yann and Bottou, Leon and Bengio, Yoshua and Haffner, Patrick},
  journal = {Proceedings of the IEEE},
  year    = {1998}
}
```

## License

[MIT](LICENSE) — free to use, modify and distribute with attribution.
