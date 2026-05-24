"""
Multi-Layer Perceptron for handwritten digit recognition (MNIST).

Architecture:  784 → 128 → 64 → 10
Activation:    sigmoid throughout (consistent with main.py)
Training:      backpropagation with mini-batch gradient descent
Data:          MNIST — fetched from Yann LeCun's site, cached locally
Animation:     live loss + accuracy curves per epoch, plus sample predictions

Run:  uv run main2.py
"""

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
#  MNIST data loader (zero extra dependencies — stdlib + numpy only)
# ──────────────────────────────────────────────────────────────────────

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = Path(__file__).parent / ".mnist_data"


def _download(url: str, dest: Path) -> None:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


def _parse_images(path: Path) -> np.ndarray:
    """Parse MNIST IDX image file → (N, 784) float64 array in [0, 1]."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    return data.astype(np.float64) / 255.0


def _parse_labels(path: Path) -> np.ndarray:
    """Parse MNIST IDX label file → (N,) int64 array."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_mnist(max_train: int = 5000, max_test: int = 1000):
    """
    Download (if needed) and load MNIST.

    Returns
    -------
    X_train : (max_train, 784)  float64
    y_train : (max_train,)       int64    labels 0–9
    X_test  : (max_test, 784)   float64
    y_test  : (max_test,)        int64
    """
    files = {k: DATA_DIR / f"{k}.gz" for k in MNIST_URLS}
    print("Loading MNIST data ...")
    for key, url in MNIST_URLS.items():
        _download(url, files[key])

    X_train = _parse_images(files["train_images"])[:max_train]
    y_train = _parse_labels(files["train_labels"])[:max_train]
    X_test  = _parse_images(files["test_images"])[:max_test]
    y_test  = _parse_labels(files["test_labels"])[:max_test]
    return X_train, y_train, X_test, y_test


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot matrix."""
    oh = np.zeros((len(labels), num_classes))
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


# ──────────────────────────────────────────────────────────────────────
#  Multi-layer neural network (sigmoid, backpropagation)
# ──────────────────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    Fully-connected feedforward network with sigmoid activation.

    Parameters
    ----------
    layer_sizes : list[int]
        e.g. [784, 128, 64, 10] → 2 hidden layers.
    """

    def __init__(self, layer_sizes: list):
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier-like init scaled for sigmoid
            w = np.random.randn(fan_in, fan_out) * np.sqrt(1.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid given *post-activation* value a."""
        return a * (1.0 - a)

    def forward(self, x: np.ndarray) -> list:
        """
        Forward pass returning all layer activations (needed for backprop).
        Returns [a0, z1, a1, z2, a2, ..., aL]
        where a0 = input, z_i = pre-activation, a_i = post-activation.
        """
        cache = [x]  # a0
        for w, b in zip(self.weights, self.biases):
            z = np.dot(cache[-1], w) + b
            a = self.sigmoid(z)
            cache.append(z)
            cache.append(a)
        return cache

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass → output activations only."""
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.sigmoid(z)
        return a

    def predict_verbose(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with detailed printouts of every internal matrix
        transformation — shows what happens step-by-step inside the network
        when identifying a digit.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        print("\n" + "=" * 70)
        print("  DIGIT IDENTIFICATION — Internal Matrix Trace")
        print("=" * 70)

        # ── Layer 0: Input ──
        print(f"\n  [LAYER 0]  Input (flattened 28×28 pixels)")
        print(f"    Shape:        {x.shape}")
        print(f"    Value range:  [{x.min():.4f}, {x.max():.4f}]")
        print(f"    Non-zero:     {np.count_nonzero(x)} / {x.size} pixels")
        # Show a mini text-based heatmap of the 28x28 digit
        img_2d = x.reshape(28, 28)
        print(f"    Digit preview (28×28, █ = pixel > 0.5):")
        for row in img_2d:
            line = "      " + "".join("██" if p > 0.3 else "░░" if p > 0.01 else "  " for p in row)
            print(line)

        a = x

        for l, (w, b) in enumerate(zip(self.weights, self.biases), start=1):
            # ── Pre-activation (weighted sum + bias) ──
            z = np.dot(a, w) + b

            # ── Post-activation (sigmoid squash) ──
            a_next = self.sigmoid(z)

            layer_type = "OUTPUT" if l == self.num_layers else f"HIDDEN {l}"
            print(f"\n  {'─' * 66}")
            print(f"  [LAYER {l}]  {layer_type}  ({w.shape[0]} → {w.shape[1]} neurons)")
            print(f"  {'─' * 66}")

            # Weight matrix summary
            print(f"    WEIGHTS  W[{l}]")
            print(f"      Shape:         {w.shape}")
            print(f"      Min / Max:     {w.min():+.4f} / {w.max():+.4f}")
            print(f"      Mean / Std:    {w.mean():+.4f} / {w.std():.4f}")

            # Bias vector summary
            print(f"    BIASES   b[{l}]")
            print(f"      Shape:         {b.shape}")
            print(f"      Min / Max:     {b.min():+.4f} / {b.max():+.4f}")

            # Pre-activation z
            print(f"    PRE-ACTIVATION  z[{l}]  (W·a + b, before sigmoid)")
            print(f"      Shape:         {z.shape}")
            print(f"      Min / Max:     {z.min():+.4f} / {z.max():+.4f}")
            if z.shape[1] <= 64:
                z_flat = z.flatten()
                # Show top 5 largest activations
                top_idx = np.argsort(z_flat)[-5:][::-1]
                print(f"      Top-5 neurons (most excited):")
                for rank, idx in enumerate(top_idx, 1):
                    bar = "█" * min(30, int(abs(z_flat[idx]) * 5))
                    print(f"        #{rank:2d}  neuron {idx:4d}  →  {z_flat[idx]:+.4f}  {bar}")
            else:
                print(f"      (layer too wide to show individual neurons)")

            # Post-activation a
            print(f"    POST-ACTIVATION a[{l}]  (after sigmoid squash → [0, 1])")
            print(f"      Shape:         {a_next.shape}")
            print(f"      Min / Max:     {a_next.min():.4f} / {a_next.max():.4f}")
            if a_next.shape[1] == 10:
                # Output layer — show all 10 digits with confidence bars
                probs = a_next.flatten()
                print(f"      Digit confidence scores:")
                for digit, prob in enumerate(probs):
                    bar_len = int(prob * 40)
                    bar = "█" * bar_len + "░" * (40 - bar_len)
                    marker = " ◀━━ PREDICTION" if digit == np.argmax(probs) else ""
                    print(f"        {digit}: {bar} {prob:.4f}{marker}")
            elif a_next.shape[1] <= 64:
                a_flat = a_next.flatten()
                print(f"      Active neurons (a > 0.5): {np.sum(a_flat > 0.5)}")
                print(f"      Inactive neurons (a < 0.1): {np.sum(a_flat < 0.1)}")

            a = a_next

        print(f"\n  {'═' * 66}")
        predicted = np.argmax(a.flatten())
        confidence = a.flatten()[predicted]
        print(f"  ★  FINAL PREDICTION:  {predicted}  (confidence: {confidence:.4f})")
        print(f"  {'═' * 66}\n")

        return a

    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray,
                   lr: float) -> float:
        """
        Single mini-batch of backpropagation. Returns batch loss (MSE).
        """
        batch_size = x_batch.shape[0]

        # ── forward pass ──
        cache = self.forward(x_batch)  # [a0, z1, a1, z2, a2, ..., aL]

        # ── output error ──
        aL = cache[-1]
        error = y_batch - aL           # (batch, 10)
        loss = np.mean(error ** 2)
        delta = error * self.sigmoid_derivative(aL)  # (batch, 10)

        # ── backpropagate ──
        deltas = [delta]  # delta for output layer (index num_layers-1)
        for l in range(self.num_layers - 1, 0, -1):
            # cache layout: [a0, z1, a1, z2, a2, ...]
            # for layer l: a_l is at position 2*l, z_l is at 2*l-1
            a_prev = cache[2 * l]      # activation of previous layer
            delta_prev = np.dot(deltas[0], self.weights[l].T) * \
                         self.sigmoid_derivative(a_prev)
            deltas.insert(0, delta_prev)

        # ── gradient descent step ──
        for l in range(self.num_layers):
            a_l = cache[2 * l]  # input to layer l
            self.weights[l] += np.dot(a_l.T, deltas[l]) * lr / batch_size
            self.biases[l] += np.sum(deltas[l], axis=0, keepdims=True) * lr / batch_size

        return float(loss)

    def train(self, X: np.ndarray, Y: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              lr: float = 0.5, X_val=None, Y_val=None):
        """
        Full training loop with live animation.

        Returns history dict with 'train_loss', 'train_acc', 'val_acc' per epoch.
        """
        n = X.shape[0]
        history = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": []}

        # ── setup live plot ──
        plt.ion()
        fig, (ax_loss, ax_acc, ax_samples) = plt.subplots(
            1, 3, figsize=(15, 5),
            gridspec_kw={"width_ratios": [1, 1, 1.3]},
        )
        fig.canvas.manager.set_window_title("MNIST Digit Classifier — Live Training")

        # loss subplot
        ax_loss.set_title("Training Loss (MSE)", fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xlim(0, epochs)
        ax_loss.set_ylim(-0.005, 0.12)
        ax_loss.grid(True, alpha=0.3)
        (line_loss,) = ax_loss.plot([], [], "C3", linewidth=1.5)

        # accuracy subplot
        ax_acc.set_title("Accuracy", fontweight="bold")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlim(0, epochs)
        ax_acc.set_ylim(-0.05, 1.05)
        ax_acc.grid(True, alpha=0.3)
        (line_train_acc,) = ax_acc.plot([], [], "C0", label="Train")
        if X_val is not None:
            (line_val_acc,) = ax_acc.plot([], [], "C1", label="Validation")
            ax_acc.legend(loc="lower right", fontsize=8)
        else:
            ax_acc.legend(loc="lower right", fontsize=8)

        # sample predictions subplot — shows 10 digits with predictions
        ax_samples.set_title("Sample Predictions", fontweight="bold")
        ax_samples.axis("off")
        sample_img = ax_samples.imshow(
            np.zeros((280, 280)), cmap="gray_r", vmin=0, vmax=1,
        )

        fig.tight_layout()

        # ── training ──
        for epoch in range(1, epochs + 1):
            # shuffle
            perm = np.random.permutation(n)
            X_shuf, Y_shuf = X[perm], Y[perm]

            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                loss = self.train_step(
                    X_shuf[start:end], Y_shuf[start:end], lr,
                )
                epoch_loss += loss * (end - start)
                num_batches += 1

            # compute metrics
            avg_loss = epoch_loss / n
            preds = self.predict(X)
            train_acc = np.mean(np.argmax(preds, axis=1) == np.argmax(Y, axis=1))

            history["epoch"].append(epoch)
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(train_acc)

            val_acc_str = ""
            if X_val is not None:
                val_preds = self.predict(X_val)
                val_acc = np.mean(np.argmax(val_preds, axis=1) == np.argmax(Y_val, axis=1))
                history["val_acc"].append(val_acc)
                val_acc_str = f"  val acc: {val_acc:.3f}"

            print(f"\rEpoch {epoch:3d}/{epochs}  "
                  f"loss: {avg_loss:.5f}  "
                  f"train acc: {train_acc:.3f}{val_acc_str}", end="")

            # update plot
            line_loss.set_data(history["epoch"], history["train_loss"])
            line_train_acc.set_data(history["epoch"], history["train_acc"])
            if X_val is not None:
                line_val_acc.set_data(history["epoch"], history["val_acc"])
            ax_loss.set_ylim(-0.005, max(0.12, avg_loss * 1.5))

            # update sample predictions: show first 10 test images + predictions
            if X_val is not None:
                display_X = X_val[:10]
                display_y = Y_val[:10]
            else:
                display_X = X[:10]
                display_y = Y[:10]
            display_preds = self.predict(display_X)
            predicted_labels = np.argmax(display_preds, axis=1)
            true_labels = np.argmax(display_y, axis=1)

            # build a 2×5 grid image
            grid = np.zeros((28 * 2, 28 * 5))
            for i in range(min(10, len(display_X))):
                r, c = i // 5, i % 5
                img = display_X[i].reshape(28, 28)
                grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = img
                # label in plot — we'll use the axis title area instead
            sample_img.set_data(grid)

            # update title with labels
            label_texts = []
            for i in range(min(10, len(display_X))):
                p = predicted_labels[i]
                t = true_labels[i]
                color = "✓" if p == t else "✗"
                label_texts.append(f"{p}({t}){color}")
            ax_samples.set_title(
                "Predictions: " + "  ".join(label_texts),
                fontweight="bold", fontsize=8, fontfamily="monospace",
            )

            plt.pause(0.01)

        print()  # newline after progress
        plt.ioff()
        plt.show(block=False)
        return history


# ──────────────────────────────────────────────────────────────────────
#  Bitmap digit loader
# ──────────────────────────────────────────────────────────────────────

def load_bitmap_digit(filepath: str, threshold: float = 0.5,
                      invert: bool | None = None) -> np.ndarray:
    """
    Load a bitmap image of a hand-drawn digit and convert it to a
    28×28 flattened array of 0/1 values ready for the neural network.

    Parameters
    ----------
    filepath : str
        Path to a PNG, JPG, BMP, or any image matplotlib can read.
    threshold : float
        Pixel intensity threshold (0–1) to binarise the image.
        Pixels above this become 1 ("ink"), below become 0 ("paper").
    invert : bool or None
        If True, assume dark digit on light background and invert.
        If None, auto-detect based on which side has more dark pixels.

    Returns
    -------
    digit : (784,) ndarray of 0.0 / 1.0
    """
    import matplotlib.image as mpimg

    print(f"\n  Loading bitmap: {filepath}")

    # Read the image
    img = mpimg.imread(filepath)
    print(f"    Original shape:  {img.shape}")

    # Convert RGBA / RGB to grayscale
    if img.ndim == 3:
        if img.shape[2] == 4:
            # RGBA — composite over white background
            alpha = img[:, :, 3:4]
            rgb = img[:, :, :3]
            img = rgb * alpha + (1 - alpha)  # blend with white
        # RGB → grayscale using luminance weights
        img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]

    # Ensure values are in [0, 1]
    if img.max() > 1.0:
        img = img / 255.0

    print(f"    Grayscale range: [{img.min():.2f}, {img.max():.2f}]")

    # Resize to 28×28 using numpy slicing / interpolation
    # We use a simple block-average resize
    h, w = img.shape
    img_resized = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            i_start = int(i * h / 28)
            i_end = int((i + 1) * h / 28)
            j_start = int(j * w / 28)
            j_end = int((j + 1) * w / 28)
            block = img[i_start:max(i_start + 1, i_end),
                       j_start:max(j_start + 1, j_end)]
            img_resized[i, j] = np.mean(block)

    print(f"    Resized to:      {img_resized.shape}")

    # Auto-detect inversion: MNIST uses white digits (1.0) on black (0.0)
    if invert is None:
        # If more than half the border is dark, we're probably dark-bg
        border_pixels = np.concatenate([
            img_resized[0, :], img_resized[-1, :],
            img_resized[1:-1, 0], img_resized[1:-1, -1],
        ])
        dark_bg = np.mean(border_pixels) < 0.5
        if dark_bg:
            # Already dark background → leave as-is (digit is bright)
            print(f"    Auto-detect:     dark background — no inversion needed")
        else:
            img_resized = 1.0 - img_resized
            print(f"    Auto-detect:     light background → INVERTED")
    elif invert:
        img_resized = 1.0 - img_resized
        print(f"    Manually inverted (dark digit → white digit)")

    # Threshold to binary
    digit = (img_resized > threshold).astype(np.float64)
    print(f"    Threshold:       {threshold}  (non-zero pixels: {np.count_nonzero(digit)})")

    # Show the processed digit as ASCII art
    print(f"    Processed digit (28×28, █ = 1, · = 0):")
    for row in digit:
        line = "      " + "".join("██" if p > 0 else "··" for p in row)
        print(line)

    return digit.flatten()


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MNIST Digit Classifier — Multi-Layer Perceptron")
    print("  Architecture: 784 → 128 → 64 → 10  (sigmoid)")
    print("=" * 60)

    # Load data (5000 train, 1000 validation from test set)
    X_train, y_train, X_test, y_test = load_mnist(max_train=5000, max_test=1000)
    Y_train = one_hot(y_train)
    Y_test = one_hot(y_test)
    print(f"  Train: {X_train.shape[0]} samples,  Test: {X_test.shape[0]} samples")
    print()

    # Build the network
    nn = NeuralNetwork([784, 128, 64, 10])
    print(f"  Layers: {784} → {128} → {64} → {10}")
    print(f"  Total parameters: {sum(w.size + b.size for w, b in zip(nn.weights, nn.biases)):,}")
    print()

    # Train with live animation
    history = nn.train(
        X_train, Y_train,
        epochs=50,
        batch_size=32,
        lr=0.5,
        X_val=X_test,
        Y_val=Y_test,
    )

    # Final evaluation
    print()
    print("=" * 60)
    test_preds = nn.predict(X_test)
    test_acc = np.mean(np.argmax(test_preds, axis=1) == y_test)
    print(f"  Final test accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print("=" * 60)

    # ── Run a verbose trace on one test sample ──
    print()
    print("=" * 60)
    print("  Verbose prediction trace (test sample 0):")
    print("=" * 60)
    sample_idx = 0
    nn.predict_verbose(X_test[sample_idx])
    print(f"  Actual label: {y_test[sample_idx]}")

    # Interactive prediction
    print()
    print("=" * 60)
    print("  Try your own digit!")
    print("=" * 60)
    print("  Options:")
    print("    [1] Enter 28 rows of pixel values manually")
    print("    [2] Load a bitmap image (PNG, JPG, BMP) of a hand-drawn digit")
    print("    [Enter] Quit")

    try:
        choice = input("\n  Choose [1/2/Enter]: ").strip()

        if choice == "1":
            print("\n  Enter 28 rows of 28 pixel values (0.0 = black bg, 1.0 = white digit):")
            print("  (Remember: MNIST uses white digits on black background!)")
            rows = []
            for i in range(28):
                row_str = input(f"    row {i:2d}: ").strip()
                rows.append([float(v) for v in row_str.split()])
            custom = np.array(rows).reshape(1, 784)
            nn.predict_verbose(custom)

        elif choice == "2":
            filepath = input("\n  Path to bitmap file: ").strip()
            if filepath and os.path.isfile(filepath):
                print("\n  Preprocessing bitmap...")
                digit = load_bitmap_digit(filepath, threshold=0.5)

                # Show the processed image using matplotlib too
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                ax1.imshow(digit.reshape(28, 28), cmap="gray_r", interpolation="nearest")
                ax1.set_title("Processed Digit (28×28 binary)", fontweight="bold")
                ax1.axis("off")

                # Predict and show bar chart
                pred = nn.predict_verbose(digit)
                probs = pred.flatten()
                ax2.bar(range(10), probs, color=["C3" if i == np.argmax(probs) else "C0" for i in range(10)])
                ax2.set_title(f"Prediction: {np.argmax(probs)} (confidence: {probs[np.argmax(probs)]:.3f})", fontweight="bold")
                ax2.set_xlabel("Digit")
                ax2.set_ylabel("Confidence")
                ax2.set_xticks(range(10))
                ax2.set_ylim(0, 1.05)
                ax2.grid(axis="y", alpha=0.3)

                fig.tight_layout()
                plt.show(block=False)
                input("\n  Press Enter to continue...")
            else:
                print(f"  File not found: {filepath}")

        else:
            print()
            print("  Goodbye!")

    except (EOFError, KeyboardInterrupt):
        print()
        print("Goodbye!")
