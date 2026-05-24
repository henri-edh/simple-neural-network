"""
Export trained network weights and per-layer activations to JSON
for the Three.js 3D visualization.

Run:  uv run export_network.py
Output:  network_data.json  (~15 MB for 5K samples)
"""

import json
import sys
from pathlib import Path

import numpy as np

# Reuse the NN code from main2.py
sys.path.insert(0, str(Path(__file__).parent))
from main2 import NeuralNetwork, load_mnist, one_hot


def export_network(json_path: str = "network_data.json",
                   max_train: int = 5000, max_test: int = 200):
    print("=" * 60)
    print("  Exporting network for 3D visualization")
    print("=" * 60)

    # ── Load data & train ──
    X_train, y_train, X_test, y_test = load_mnist(
        max_train=max_train, max_test=max_test,
    )
    Y_train = one_hot(y_train)
    Y_test = one_hot(y_test)

    nn = NeuralNetwork([784, 128, 64, 10])
    print(f"  Training on {len(X_train)} samples ...")
    history = nn.train(
        X_train, Y_train,
        epochs=30,
        batch_size=32,
        lr=0.5,
        X_val=X_test,
        Y_val=Y_test,
    )

    test_preds = nn.predict(X_test)
    test_acc = np.mean(np.argmax(test_preds, axis=1) == y_test)
    print(f"\n  Test accuracy: {test_acc:.4f}")

    # ── Export: weights & biases ──
    weights_export = []
    for l, (w, b) in enumerate(zip(nn.weights, nn.biases)):
        weights_export.append({
            "layer": l,
            "weights": w.tolist(),
            "bias": b.flatten().tolist(),
            "shape_in": w.shape[0],
            "shape_out": w.shape[1],
        })

    # ── Export: per-sample activations ──
    samples = []
    for i in range(len(X_test)):
        # Full forward pass with cache
        cache = nn.forward(X_test[i:i+1])

        # Extract activations from cache: [a0, z1, a1, z2, a2, ..., aL]
        layers = []
        # a0 (input)
        layers.append({
            "name": "input",
            "type": "activation",
            "shape": [28, 28],
            "values": X_test[i].tolist(),
        })

        for l in range(nn.num_layers):
            z_idx = 2 * l + 1  # z
            a_idx = 2 * l + 2  # a
            z_vals = cache[z_idx].flatten().tolist()
            a_vals = cache[a_idx].flatten().tolist()

            layer_name = "output" if l == nn.num_layers - 1 else f"hidden{l+1}"
            layers.append({
                "name": layer_name,
                "type": "pre_activation",
                "shape": [nn.weights[l].shape[1]],
                "values": z_vals,
            })
            layers.append({
                "name": layer_name,
                "type": "activation",
                "shape": [nn.weights[l].shape[1]],
                "values": a_vals,
            })

        samples.append({
            "index": i,
            "label": int(y_test[i]),
            "prediction": int(np.argmax(test_preds[i])),
            "correct": bool(y_test[i] == np.argmax(test_preds[i])),
            "layers": layers,
        })

    # ── Build final JSON ──
    data = {
        "architecture": {
            "layers": [784, 128, 64, 10],
            "activation": "sigmoid",
            "num_samples": len(samples),
            "test_accuracy": float(test_acc),
        },
        "weights": weights_export,
        "samples": samples,
    }

    print(f"\n  Writing {json_path} ...")
    with open(json_path, "w") as f:
        json.dump(data, f)

    size_mb = Path(json_path).stat().st_size / (1024 * 1024)
    print(f"  Done!  {json_path}  ({size_mb:.1f} MB)")
    print(f"  {len(samples)} samples exported with full layer activations")


if __name__ == "__main__":
    export_network()
