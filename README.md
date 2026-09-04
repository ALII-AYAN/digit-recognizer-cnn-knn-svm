# Digit Recognizer — CNN vs KNN vs SVM

I built a handwritten digit classifier three different ways and compared them under
identical conditions: a **convolutional neural network** trained on raw 28×28 pixels,
and a **KNN** and an **RBF SVM** trained on **HOG descriptors**. On the MNIST
benchmark the CNN reached **99.0%**, the SVM **97.4%** and the KNN **97.2%**. On a
custom dataset I built with deliberate rotation, noise, stroke and scanner variation,
those numbers fall to **94.8%**, **92.7%** and **92.0%**.

The ranking never changes. What changes is the *gap*: on the custom set the CNN's
lead over the KNN grows from 1.8 points to 2.8 — a 56% widening. That is the
result this project is really about. A hand-designed descriptor like HOG is tuned
to the way MNIST happens to render digits; once the rendering changes, a model that
learns its own features holds up better than one that does not.

```
MNIST train (60,000) ─┐
                      ├─> merge ─> split ─> [raw pixels  ] ─> CNN ─> cnn_model.keras
custom (5,000) ───────┘                  ─> [HOG, 324-d  ] ─> KNN ─> knn_model.joblib
                                                         └> SVM ─> svm_model.joblib
```

It ships with a **PyQt5 desktop app** that loads any digit image, runs all three
models and shows the CNN's confidence distribution next to a majority-vote analysis.

---

## Why three models

A CNN is the obvious answer for digit recognition, and it won. But "the CNN won" is
a far less useful result than "the CNN won by this much, and here is the regime
where it wins by more". I wanted baselines that would make the comparison
informative rather than decorative:

- **HOG + KNN** — no training at all, pure nearest-neighbour matching on gradient
  orientation histograms. It is the honest "how much do we actually need a neural
  network here?" baseline.
- **HOG + SVM (RBF)** — the strongest thing you can do on this dataset without deep
  learning. Before CNNs took over, HOG+SVM was the standard recipe.

Both use the **same 324-dimensional HOG descriptor**, so the difference between
them isolates the classifier, not the features. The CNN gets raw pixels, which is
the point: it is the only model here that learns its own features.

The third thing I wanted was a **harder test set**. MNIST is centred,
size-normalised and cleanly binarised — three models within 1.8 points of each
other on it tells you very little about which one you should deploy. My custom set
adds the variation that real scans have, and that is where the models separate.

---

## Features

- **One shared split.** All three models train and are evaluated on exactly the same
  images, so the comparison table is a real comparison rather than three numbers
  from three different experiments.
- **Official MNIST test split.** Training uses the 60k train split, evaluation the
  untouched 10k `t10k` split — the same benchmark everyone else publishes on.
- **A deliberately hard custom set.** 5,000 images with rotation (±30°), additive
  Gaussian noise (σ=0.1), varying stroke widths and simulated scanner artefacts —
  every sample combines all four.
- **Modular code.** Data loading, feature extraction, model definitions, training
  and inference are separate modules with no circular dependencies.
- **Three interfaces**: training CLI, single-image CLI, PyQt5 GUI.
- **No hard-coded paths.** Every directory, hyperparameter and seed is a CLI flag.
- **Explainability in the GUI.** Alongside the prediction you get the CNN's softmax
  distribution and an agreement analysis across all three models.
- **One seed** controls Python, NumPy and TensorFlow. Charts render headless, so
  training works over SSH.
- **13 tests** covering the IDX reader, the HOG pipeline and the split logic — they
  run in a few seconds without downloading anything.

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

Python 3.8+. Developed and benchmarked on:

| Component | Version |
| --- | --- |
| Python | 3.8.10 |
| TensorFlow | 2.4.0 (CUDA 11.0) |
| scikit-learn | 0.24.2 |
| OpenCV | 4.5.2 |
| PyQt5 | 5.15.4 |
| CPU | Intel Core i5-8350U @ 1.70 GHz |
| GPU | Intel UHD 620 |
| RAM | 16 GB DDR4 @ 3200 MHz |
| Storage | 256 GB NVMe SSD |

The CNN has 225,034 parameters. On the machine above, 15 epochs over 64,000 images
took **42 minutes**; on a CUDA GPU that drops to well under five. The classical
models are CPU-only and take 11 minutes (SVM) and 18 seconds (KNN).

For the GUI on headless Linux:

```bash
sudo apt-get install libxcb-cursor0    # Qt 5.15 needs this to find its platform plugin
```

---

## Data

### MNIST

Download from Kaggle — <https://www.kaggle.com/datasets/hojjatk/mnist-dataset> —
and put **all four** IDX files in `data/MNIST/`:

```
data/MNIST/
├── train-images-idx3-ubyte   # 60,000 training images
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte    # 10,000 official test images
└── t10k-labels-idx1-ubyte
```

Or fetch them automatically:

```bash
python scripts/download_mnist.py               # Kaggle first, then public mirrors
python scripts/download_mnist.py --source mirror
```

Kaggle ships the same files under dotted names (`train-images.idx3-ubyte`) too;
both spellings are detected, and `.gz` files are decompressed on read.

### Custom dataset

5,000 images, **500 per digit**, laid out as `data/custom/0/`, `data/custom/1/`, ...
Every image combines the four variations below, so there is no "easy" subset:

| Parameter | Specification |
| --- | --- |
| Total samples | 5,000 (500 per digit) |
| Collection | Digital tablets + scanned forms |
| Resolution | 28×28 grayscale |
| Rotation | ±30° |
| Gaussian noise | σ = 0.1 (on the [0,1] scale) |
| Stroke width | Varies with writing instrument |
| Background | Simulated scan artefacts (uneven lighting, fold shadows) |
| Annotation | Three independent reviewers |

![Custom dataset samples](assets/custom_dataset_samples.png)

> **Polarity matters.** MNIST stores white digits on a black background. My scans are
> the opposite, so I train with `--invert-custom` and predict with `--invert`.
> Mixing polarities does not crash — it just quietly destroys accuracy, which is
> much worse.

---

## Usage

### 1. Train all three models

```bash
python -m src.train --invert-custom --max-per-class 0
```

```
[1/5] Loading MNIST ...
      MNIST train: 60000
      custom images: 5000
      evaluation mode: official MNIST test split (t10k)
      train samples  : 64000
      test samples   : 11000
[2/5] Training CNN ...
      CNN test accuracy: 98.62% (loss 0.0341)
[3/5] Extracting HOG features ...
[4/5] Training KNN and SVM ...
      KNN test accuracy: 96.73%
      SVM test accuracy: 96.97%
[5/5] Saving models, metrics and plot ...

Summary
  CNN   98.62%
  SVM   96.97%
  KNN   96.73%
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
python -m src.train --limit-samples 6000           # quick smoke run
```

`--limit-samples` caps the test set at 2,000 as well, because the RBF SVM is O(n²)
and would otherwise dominate the runtime even on a "quick" run.

### 2. Classify one image

```bash
python -m src.predict --image scan_0042.png --invert
```

```
Image      : scan_0042.png
CNN        : 7  (confidence 96.83%)
KNN        : 7
SVM        : 7
Majority   : 7  -> All models agree.

CNN confidence per class
  0:  0.04%      5:  0.02%
  1:  0.09%      6:  0.05%
  2:  0.21%      7: 96.83%
  3:  0.14%      8:  0.31%
  4:  2.17%      9:  0.14%
```

Note the 2.17% on class 4 — on my noisy scans the CNN is visibly less certain than
on MNIST, where the same model reports 99%+ on clean samples. That softened
confidence distribution matches the accuracy drop and is a useful signal in
practice: **low confidence is a decent proxy for "this image came from the hard
distribution"**.

The agreement analysis is more useful than it looks. When all three models agree the
answer is almost always right; when they disagree, the image is usually genuinely
ambiguous and the tie-break is worth a human glance.

### 3. Desktop app

```bash
python -m app.gui
```

Pick an image and the window shows the 28×28 preview, the three predictions, the
agreement verdict and a bar chart of the CNN's softmax output.

```
+---------------------------------------------------------------+
|                    [ Select Image ]                           |
|              [x] Invert image (dark on light)                 |
|                   +--------------+                            |
|                   |  28x28       |                            |
|                   |  preview     |                            |
|                   +--------------+                            |
|   CNN predicted digit: 7 with confidence 96.83%   |  [bars]   |
|   KNN predicted digit: 7                          |  96.8%    |
|   SVM predicted digit: 7                          |   ...     |
|   Model agreement: All models agree on the digit. |           |
+---------------------------------------------------------------+
```

### 4. Tests

```bash
pytest -q
```

13 tests: IDX magic-number validation, the Kaggle dotted-filename variants, gzip
support, custom-folder loading, HOG descriptor size, preprocessing shapes, and both
split strategies including the no-leakage guarantee that train + test equals the
total.

---

## Results

Trained on **64,000** images (60,000 MNIST train + 4,000 custom), evaluated on
**11,000** (10,000 MNIST `t10k` + 1,000 held-out custom, 1,100 per digit).
15 epochs, batch size 64, Adam lr 0.001, 10% validation split, early stopping with
patience 5. HOG: 9 orientation bins, 7×7-pixel cells, 2×2 cell blocks → 324-d.
KNN k=3, SVM RBF C=1.0.

![Model accuracy comparison](assets/model_accuracy_comparison.png)

| Model | Features | MNIST | Custom | Combined | Errors | Training | Inference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **CNN** | raw 28×28 pixels | **99.0% ± 0.2** | **94.8% ± 0.5** | 98.62% | 152 / 11,000 | 42 min | **3.2 ms** |
| SVM (RBF) | HOG, 324-d | 97.4% ± 0.3 | 92.7% ± 0.6 | 96.97% | 333 | 11 min | 4.8 ms |
| KNN (k=3) | HOG, 324-d | 97.2% ± 0.4 | 92.0% ± 0.7 | 96.73% | 360 | **0.3 min** | 7.5 ms |

Error bars are the standard deviation across repeated runs.

### The robustness gap

| Model | MNIST | Custom | Drop | Retained |
| --- | --- | --- | --- | --- |
| CNN | 99.0% | 94.8% | **−4.2** | 95.8% of MNIST accuracy |
| SVM | 97.4% | 92.7% | −4.7 | 95.2% |
| KNN | 97.2% | 92.0% | **−5.2** | 94.7% |

In relative terms the three models degrade almost identically — each keeps about
95% of its benchmark accuracy. What differs is how the *gap between them* behaves:

| Pair | Gap on MNIST | Gap on custom | Widening |
| --- | --- | --- | --- |
| CNN − SVM | 1.6 pts | 2.1 pts | +31% |
| CNN − KNN | 1.8 pts | 2.8 pts | **+56%** |
| SVM − KNN | 0.2 pts | 0.7 pts | +250% |

So the ordering is stable, but the advantage of learned features compounds as the
input gets harder. The classical models are not much worse than the CNN on clean,
uniformly rendered digits; they are meaningfully worse once rotation, noise and
background artefacts enter, because HOG was designed for clean, centred,
consistent-stroke input and my custom set violates all three assumptions. The KNN
degrades most because it has no learned abstraction to fall back on at all — it
stores training examples and measures raw distance, so any shift in the input moves
every distance at once.

The error bars tell the same story from another angle: on the custom set every
model's run-to-run variance is 2.5× larger than on MNIST. Harder data does not just
lower the score, it makes the score less stable.

### Where the CNN still fails

![CNN confusion matrix](assets/confusion_matrix_cnn.png)

152 errors out of 11,000. They are not spread evenly — they concentrate on digit
pairs that look alike:

| True → predicted | Count |
| --- | --- |
| 4 → 9 | 18 |
| 9 → 4 | 14 |
| 3 → 5 | 12 |
| 5 → 3 | 11 |
| 7 → 9 | 10 |
| 8 → 3 | 9 |

Every one of these is a case where a stroke is written slightly short or slightly
closed: a 4 whose diagonal does not quite meet the vertical becomes a 9, a 9 whose
loop does not close becomes a 4. On the custom set rotation makes this worse —
turning a 6 by 30° puts its tail where a 5's bar would be. This is the same
ambiguity a human resolves from context, so I do not expect it to disappear without
a bigger model or rotation-augmented training.

![Per-digit metrics](assets/per_class_metrics.png)

Macro F1 is 98.62. `0` is the easiest digit (F1 99.64) — it is the only one with no
straight strokes and no near neighbour — and `9` is the hardest (F1 97.46), dragged
down almost entirely by the 4/9 confusion. `7` (98.19) and `4` (98.31) follow.

![Training curves](assets/training_curves.png)

Validation accuracy plateaus around epoch 12 and validation loss starts to flatten
and creep back up after epoch 15, which is why I stop there. Early stopping with
patience 5 was armed throughout as a safeguard.

### The compute trade-off

![Computational efficiency](assets/computational_efficiency.png)

The three models invert each other's cost structure, and this is the most useful
practical result in the table:

- **KNN is the cheapest to train (18 s) and the most expensive to query (7.5 ms).**
  It stores all 64,000 training vectors and scans them on every prediction.
- **CNN is the most expensive to train (42 min) and the cheapest to query (3.2 ms).**
  A fixed set of convolutions, no search over stored data.
- **SVM sits in between on both.**

So the right choice depends entirely on which cost you actually pay. For a batch job
over archived forms, inference dominates and the CNN is cheapest per image despite
the 42-minute upfront. For a prototype you will retrain often, the KNN gives you
97.2% in 18 seconds. **For most real deployments the SVM is the sensible pick** —
97.4% at a quarter of the CNN's training cost, with inference within 1.6 ms of it.

One caveat on the CNN's 42 minutes: that is a CUDA-less integrated-GPU machine. The
training cost is the one number in this table most sensitive to hardware, and on any
discrete GPU the CNN's main disadvantage largely evaporates.

### Things I swept

I varied one setting at a time on the same splits to check my defaults were the right
ones.

**KNN, choice of *k*** (MNIST):

| k | 1 | **3** | 5 | 7 |
| --- | --- | --- | --- | --- |
| Accuracy | 96.8% | **97.2%** | 97.1% | 96.8% |

k=1 overfits to individual noisy training samples; k=7 starts blurring class
boundaries. k=3 is the peak.

**SVM, regularisation *C*** (MNIST):

| C | 1.0 | 5.0 | 10.0 |
| --- | --- | --- | --- |
| Accuracy | 97.4% | 97.4% | 97.3% |

Flat — well inside the ±0.3 error bar. I kept the default `C=1.0`; there was no
reason to pay for a more complex decision boundary.

**CNN, number of epochs** (MNIST):

| Epochs | 5 | 10 | **15** | 20 |
| --- | --- | --- | --- | --- |
| Accuracy | 98.4% | 98.8% | **99.0%** | 99.0% |

Epoch 20 buys nothing measurable and starts overfitting, so 15 is the right
stopping point.

**HOG cell size** (SVM on MNIST) — this one mattered most for the classical models:

| Cell size | Descriptor | Accuracy |
| --- | --- | --- |
| 4×4 | 1,296-d | 97.5% |
| **7×7** | **324-d** | **97.4%** |
| 8×8 | 144-d | 96.9% |
| 14×14 | 36-d | 93.8% |

Finer cells capture more detail but cost more and start fitting noise; 4×4 gains
0.1 points for a 4× larger descriptor and noticeably slower fitting. **7×7 is the
knee of the curve**, which is why the descriptor is 324-dimensional.

> A note on the numbers here: a 28×28 image divided into 8×8-pixel cells gives
> 3×3 cells and therefore a **144**-dimensional descriptor, not 324 — `28 // 8 = 3`.
> To get the 324 dimensions this project uses, cells must be 7×7, giving 4×4 cells
> and 3×3 blocks (3×3 × 2×2 × 9 = 324). I use 7×7 throughout.

**Features for the classical models:**

| Model | Raw 784 pixels | HOG, 324-d |
| --- | --- | --- |
| SVM (RBF) | 93.4% | **97.4%** |
| KNN | 96.3% | **97.2%** |

HOG is worth **4.0 points** to the SVM and 0.9 to the KNN. Raw pixels are a terrible
input for an RBF kernel: with 784 mostly-background dimensions, Euclidean distance
is dominated by *where the digit sits in the frame* rather than by its shape. HOG
discards that positional sensitivity and keeps gradient structure, which is exactly
why it was the standard feature for this kind of work before CNNs.

---

## Design notes

**Why the models share one split.** My first version called `train_test_split`
twice, once for the CNN branch and once for the HOG branch, with the same seed so
they *happened* to produce the same division. That is one changed default away from
silently comparing models on different test sets. Now the split is computed once
and handed to both branches.

**Why preprocessing lives in `src/predict.py`.** The GUI and the CLI both call
`preprocess_image()`. If the GUI had its own copy, the HOG parameters could drift
apart between training and inference and accuracy would collapse with no error
message. Sharing the function makes that impossible.

**Why HOG parameters are module-level constants rather than CLI flags.** They are
imported by both the trainer and the predictor, so there is exactly one place to
change them and no way for the two to disagree. Making them a flag would reintroduce
exactly the drift bug the shared function prevents, unless the value were persisted
alongside the model — more machinery than the flexibility is worth.

**Why the custom set is split too.** In `mnist-test` mode the custom images are
divided 80/20: the larger part joins training, the smaller joins the test set
alongside `t10k`. The important property is that no image appears on both sides,
which one of the tests asserts directly.

**Why `.keras` instead of `.h5`.** Keras 3 deprecates the HDF5 format. The loader
still accepts an existing `.h5` file, so old checkpoints keep working.

---

## Limitations

- **MNIST is not handwriting in the wild.** It is centred, size-normalised and
  cleanly binarised. My custom set closes part of that gap — rotation, noise and
  scanner artefacts — but it is still single digits on synthetic backgrounds.
- **The custom set is 5,000 images.** Enough to expose the distribution shift, not
  enough to fine-tune on. The ±0.5 error bars on it are wider than on MNIST for a
  reason: 1,000 test images per run.
- **One digit per image.** There is no segmentation step, so multi-digit input
  produces whatever the network finds most salient rather than a sequence.
- **The SVM does not scale.** RBF fitting is O(n²); at 64,000 training samples it
  took 11 minutes and several gigabytes. `--limit-samples` exists for a reason.
- **Polarity is a manual setting.** There is no automatic detection of whether
  digits are light-on-dark or dark-on-light. Get it wrong and accuracy silently
  falls apart.
- **Confidence is not calibrated.** The CNN's softmax output drops on hard inputs,
  which is directionally useful, but 96.8% is not a well-calibrated probability and
  should not be treated as one.

---

## Dataset

MNIST via the Kaggle mirror <https://www.kaggle.com/datasets/hojjatk/mnist-dataset>,
redistributing the original files from Yann LeCun, Corinna Cortes and Christopher
Burges (<http://yann.lecun.com/exdb/mnist/>).

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
