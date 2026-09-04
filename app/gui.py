"""PyQt5 desktop app: pick an image, see CNN / KNN / SVM predictions side by side.

Run from the project root:

    python -m app.gui
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow both `python -m app.gui` (from the project root) and `python app/gui.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtGui import QFont, QImage, QLinearGradient, QBrush, QPalette, QPixmap, QColor  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.predict import load_models, preprocess_image  # noqa: E402


class MplCanvas(FigureCanvas):
    """Embedded Matplotlib bar chart showing CNN softmax confidences."""

    def __init__(self, width: int = 5, height: int = 3, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.fig.patch.set_facecolor("#f0f0f0")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def plot_confidences(self, confidences) -> None:
        self.ax.clear()
        bars = self.ax.bar(range(10), confidences, color="#3498db")
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks(range(10))
        self.ax.set_xticklabels(str(i) for i in range(10))
        self.ax.set_ylabel("Confidence")
        self.ax.set_title("CNN Prediction Confidence")
        for bar, conf in zip(bars, confidences):
            self.ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{conf:.2f}",
                ha="center",
                fontsize=8,
            )
        self.draw()


class DigitRecognizerGUI(QWidget):
    def __init__(self, models_dir: Path | None = None):
        super().__init__()
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / "models"
        self.models = None
        # Keep a reference to the numpy buffer behind the QImage alive.
        self._current_image: np.ndarray | None = None

        self.setWindowTitle("Digit Recognizer - CNN, KNN & SVM")
        self.setGeometry(150, 150, 850, 680)
        self.setMinimumSize(850, 680)

        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#2c3e50"))
        gradient.setColorAt(1.0, QColor("#3498db"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)

        self.btn_select = QPushButton("Select Image")
        self.btn_select.setFixedHeight(40)
        self.btn_select.setStyleSheet(
            """
            QPushButton {
                background-color: #2980b9; color: white; font-size: 16px; border-radius: 8px;
            }
            QPushButton:hover { background-color: #1abc9c; }
            QPushButton:disabled { background-color: #7f8c8d; }
            """
        )
        self.btn_select.clicked.connect(self.open_file_dialog)

        self.invert_checkbox = QCheckBox("Invert image (black digits on white background)")
        self.invert_checkbox.setStyleSheet("QCheckBox { color: white; font-size: 13px; }")

        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white; border: 2px solid #2980b9;")
        self.image_label.setFixedSize(250, 250)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 12))
        self.result_text.setStyleSheet(
            "background-color: #ecf0f1; color: #2c3e50; padding: 10px; border-radius: 8px;"
        )
        self.result_text.setFixedWidth(380)

        self.canvas = MplCanvas(width=5, height=3, dpi=100)

        bottom = QHBoxLayout()
        bottom.addWidget(self.result_text)
        bottom.addWidget(self.canvas)

        layout = QVBoxLayout()
        layout.addWidget(self.btn_select, alignment=Qt.AlignCenter)
        layout.addWidget(self.invert_checkbox, alignment=Qt.AlignCenter)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addLayout(bottom)
        self.setLayout(layout)

        self.load_models_async()

    # ------------------------------------------------------------------ setup
    def load_models_async(self) -> None:
        try:
            self.models = load_models(self.models_dir)
        except SystemExit as exc:  # raised by load_models when files are missing
            self.btn_select.setEnabled(False)
            self.result_text.setText(f"{exc}\n\nTrain the models first:\n    python -m src.train")
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self.btn_select.setEnabled(False)
            self.result_text.setText(f"Failed to load models:\n{exc}")

    # ------------------------------------------------------------- interaction
    def open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Digit Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
        )
        if path:
            self.process_image(path)

    def process_image(self, image_path: str) -> None:
        invert = self.invert_checkbox.isChecked()
        cnn_input, hog_feature, resized = preprocess_image(image_path, invert=invert)
        if cnn_input is None:
            self.result_text.setText("Error: could not load image.")
            self.image_label.setText("No image selected")
            return

        self._current_image = resized  # keep the pixel buffer alive for QImage
        height, width = resized.shape
        qimg = QImage(resized.data, width, height, resized.strides[0], QImage.Format_Grayscale8)
        self.image_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        cnn_probs = self.models["CNN"].predict(cnn_input, verbose=0)[0]
        cnn_pred = int(np.argmax(cnn_probs))
        cnn_conf = float(cnn_probs[cnn_pred])
        knn_pred = int(self.models["KNN"].predict(hog_feature)[0])
        svm_pred = int(self.models["SVM"].predict(hog_feature)[0])

        votes = [cnn_pred, knn_pred, svm_pred]
        majority = max(set(votes), key=votes.count)
        agreement = votes.count(majority)
        if agreement == 3:
            analysis = "All models agree on the digit."
        elif agreement == 2:
            analysis = f"Two models agree on digit {majority}."
        else:
            analysis = "Models disagree on predictions."

        self.result_text.setText(
            f"CNN predicted digit: {cnn_pred} with confidence {cnn_conf * 100:.2f}%\n"
            f"KNN predicted digit: {knn_pred}\n"
            f"SVM predicted digit: {svm_pred}\n\n"
            f"Model agreement analysis:\n{analysis}\n\n"
            "Note: CNN confidence is the softmax output.\n"
            "KNN & SVM predictions are discrete labels."
        )
        self.canvas.plot_confidences(cnn_probs)


def main() -> None:
    app = QApplication(sys.argv)
    window = DigitRecognizerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
