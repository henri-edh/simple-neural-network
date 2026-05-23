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

    # Interactive prediction
    print()
    print("Try your own digit! Enter 28 lines of 28 space-separated pixel values")
    print("(0.0–1.0), or press Enter to skip.")
    try:
        choice = input("Enter 'y' to input a custom digit, or Enter to quit: ").strip()
        if choice.lower() == "y":
            print("Enter 28 rows of 28 pixel values (0.0 = white, 1.0 = black):")
            rows = []
            for i in range(28):
                row_str = input(f"  row {i:2d}: ").strip()
                rows.append([float(v) for v in row_str.split()])
            custom = np.array(rows).reshape(1, 784)
            pred = nn.predict(custom)
            print(f"  Prediction: digit {np.argmax(pred)}")
            print(f"  Confidence scores: {pred.flatten().round(4)}")
    except (EOFError, KeyboardInterrupt):
        print()
        print("Goodbye!")
