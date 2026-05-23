# Architecture: Simple Neural Network

## Overview

This project implements a **single-layer perceptron** — the simplest possible neural network architecture — in Python using NumPy. It consists of an input layer with 3 neurons connected directly to 1 output neuron, with **no hidden layers**. The network learns to predict a binary output from 3 binary inputs using supervised learning via gradient descent.

The repository contains two implementations of the same algorithm:
- **`training_version.py`** — A fully procedural, linear script designed to teach the mechanics step-by-step.
- **`main.py`** — An object-oriented, reusable `NeuralNetwork` class that can be instantiated and trained on arbitrary data.

---

## Project Structure

```
simple-neural-network/
├── main.py                # OOP NeuralNetwork class (importable, interactive, animated)
├── training_version.py    # Procedural training script (educational, one-shot)
├── pyproject.toml          # uv project config & dependencies
├── .python-version         # Pin Python 3.12
├── outputs.txt            # Terminal output of training_version.py
├── README.md              # Project description and quickstart
└── architecture.md        # This file
```

---

## Architectural Components

### 1. Network Topology

```
 Input Layer          Output Layer
 (3 neurons)          (1 neuron)
                                    
   [I1] ──w1──┐                     
               │                     
   [I2] ──w2──┼── Σ ──► sigmoid ──► Output (0–1)
               │                     
   [I3] ──w3──┘                     
```

- **Inputs**: 3 binary values (`I1`, `I2`, `I3`)
- **Weights**: A 3×1 matrix (`w1`, `w2`, `w3`), initialized randomly in range [-1, 1] with mean 0
- **Summation**: Weighted sum: `z = I1·w1 + I2·w2 + I3·w3`
- **Activation**: Sigmoid function squashes `z` into range (0, 1)
- **Output**: A probability-like value; thresholded at 0.5 for binary classification

### 2. Activation Function: Sigmoid

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

The sigmoid maps any real-valued input into a smooth S-curve between 0 and 1, making it suitable for binary output interpretation.

### 3. Training Algorithm: Gradient Descent (Delta Rule)

The network is trained via **supervised learning**. For each of 10,000 iterations the following occurs:

1. **Forward pass** — inputs are multiplied by the synaptic weights and passed through sigmoid to produce a prediction.
2. **Error calculation** — the difference between the training output and the predicted output: `error = true_output - predicted_output`
3. **Gradient computation** — the error is multiplied by the sigmoid derivative evaluated at the current output. The derivative `σ'(x) = x·(1 - x)` determines *how much* each weight should be adjusted. Weights that produced uncertain outputs (near 0.5) receive larger adjustments.
4. **Weight update** — the adjustments are computed as the dot product of the transposed inputs with the weighted error, and the synaptic weights are updated: `weights += adjustments`

This is effectively the **delta rule** (or stochastic gradient descent for a single-layer network), which converges toward weights that minimize the squared error.

### 4. Training Data

| Input 1 (`I1`) | Input 2 (`I2`) | Input 3 (`I3`) | Expected Output |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 1 | **0** |
| 1 | 1 | 1 | **1** |
| 1 | 0 | 1 | **1** |
| 0 | 1 | 1 | **0** |

**Pattern**: The output is `1` if and only if Input 1 is `1`. Inputs 2 and 3 are effectively noise/distractors. Input 3 is always `1` across all examples (acting as a "bias-like" feature).

### 5. Learned Weights (after 10,000 iterations)

```
w1 =  9.67   (strong positive — Input 1 drives output to 1)
w2 = -0.21   (small negative — Input 2 has little influence)
w3 = -4.63   (strong negative — constant Input 3 biases toward 0)
```

The network successfully learned: when `I1=1`, the large positive `w1` dominates and produces output near `1`; when `I1=0`, the large negative `w3` plus the bias of the sigmoid pushes toward near `0`.

---

## Implementation Details

### `main.py` — NeuralNetwork Class

| Method | Description |
|---|---|
| `__init__()` | Seeds the RNG for reproducibility. Initializes `synaptic_weights` as a random 3×1 matrix in [-1, 1]. |
| `sigmoid(x)` | Static method. Applies the sigmoid activation element-wise. |
| `sigmoid_derivative(x)` | Static method. Returns `x * (1 - x)`, the gradient of sigmoid. |
| `think(inputs)` | Forward pass. Takes a numpy array, casts to float, computes `sigmoid(dot(inputs, weights))`. |
| `train(inputs, outputs, iterations)` | Runs the training loop for `iterations` steps as described above. Updates `self.synaptic_weights` in-place. |
| `train_with_animation(inputs, outputs, iterations, sample_interval=50)` | Same training logic as `train()`, but renders a live 3-panel matplotlib dashboard that animates weights, predictions, and error in real time. |

The `__main__` guard provides an interactive demo: it trains on the 4-example dataset for 10,000 iterations, prints the initial and final weights, then prompts the user for 3 binary inputs and prints the predicted output.

### `training_version.py` — Procedural Script

The same logic as `main.py`, but flattened into a single script with no classes or functions beyond `sigmoid` and `sigmoid_derivative`. All variables are module-level. It prints:
- Random starting weights
- Final trained weights
- The network's predictions on the training set after training

Its output is captured in `outputs.txt`.

---

## Data Flow

```
                        ┌──────────────────┐
                        │  Random Seed (1)  │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Initialize      │
                        │  Weights (3×1)   │
                        │  range [-1, 1]   │
                        └────────┬─────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
     ┌────────▼────────┐ ┌──────▼──────┐  ┌────────▼────────┐
     │ Training Inputs  │ │  Forward    │  │ Training Outputs │
     │ (4×3 matrix)     │ │  Pass       │  │ (4×1 matrix)     │
     └────────┬────────┘ │  think()    │  └────────┬────────┘
              │          └──────┬──────┘           │
              │                 │                  │
              │          ┌──────▼──────┐           │
              └──────────►  Predicted  ◄───────────┘
                         │  Output     │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  error =    │
                         │  true - pred│
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │ adjustments │
                         │ = dot(T_in, │
                         │   err * σ') │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │ weights +=  │
                         │ adjustments │
                         └──────┬──────┘
                                │
                         ╔══════╧══════╗
                         ║  Repeat     ║
                         ║  10,000×    ║
                         ╚═════════════╝
```

---

## Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| **No hidden layers** | Keeps the network trivially simple for educational purposes — this is a perceptron. |
| **Sigmoid over ReLU/tanh** | Sigmoid's output range (0,1) directly maps to binary classification. Its smooth derivative enables gradient descent. The original tutorial authors chose it for clarity. |
| **NumPy for matrix math** | Vectorized `np.dot()` is concise and performant for the small 4×3 training matrix. No need for a deep learning framework. |
| **Fixed random seed (1)** | Ensures reproducible results across runs. The specific seed produces weights that converge cleanly on the target pattern. |
| **10,000 iterations** | Empirically sufficient for the weights to converge on this simple 4-example dataset. |
| **No bias term** | The architecture uses only weights, not an explicit bias neuron. Input 3 (always `1`) effectively serves as a learned bias. |
| **Procedural + OOP dual implementation** | `training_version.py` serves as the "whiteboard explanation" — no abstractions, every step visible. `main.py` is the "tool" — encapsulated, reusable, interactive. |

---

## Tooling

This project uses **[uv](https://docs.astral.sh/uv/)** (v0.11+) for Python version management, virtual environments, and dependency resolution. Everything is declared in `pyproject.toml` under `[project].dependencies`.

```bash
uv run main.py          # auto-creates venv, installs deps, runs the script
uv run training_version.py
uv sync                 # install deps into .venv
uv add <package>        # add a new dependency
```

Python 3.12 is pinned via `.python-version`.

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24 | Matrix multiplication (`np.dot`), exponential (`np.exp`), random number generation |
| `matplotlib` | ≥3.7 | Real-time animated training dashboard (interactive mode, subplots, legends) |

---

## Live Training Animation

The `train_with_animation()` method renders a real-time 3-panel matplotlib dashboard during training:

```
┌──────────────────────────────────────────────────┐
│  Subplot 1: Synaptic Weights                     │
│  Three curves (w₁, w₂, w₃) converging from       │
│  random initialization to final learned values   │
├──────────────────────────────────────────────────┤
│  Subplot 2: Predicted Outputs vs Targets         │
│  Four solid curves approaching dashed target     │
│  lines at y=0 and y=1 as training progresses     │
├──────────────────────────────────────────────────┤
│  Subplot 3: Mean Squared Error                   │
│  MSE decaying toward zero on a log-like curve    │
└──────────────────────────────────────────────────┘
```

### Timing Design

| Parameter | Value | Rationale |
|---|---|---|
| Total iterations | 10,000 | Sufficient for convergence on this dataset |
| Sample interval | 50 | 10,000 ÷ 50 = 200 frames |
| Frames rendered | 200 | Smooth animation without over-sampling |
| `plt.pause()` | 0.001 s | Minimal overhead; the OS scheduler throttles actual refresh to ~20–30 fps |
| Perceived duration | ~8–12 s | Feels responsive and watchable |

Sampling every 50 iterations (not every single one) avoids the overhead of 10,000 matplotlib redraws while still producing enough frames for a fluid curve animation. The final iteration is always recorded (even if it doesn't land on a sample boundary) to show the converged state.

## Limitations

1. **Single-layer only** — Cannot learn non-linearly-separable patterns (XOR problem).
2. **Fixed topology** — The 3→1 neuron structure is hardcoded. No way to add layers or change input/output dimensions without editing source.
3. **No batching** — The entire training set is processed at once (full-batch gradient descent).
4. **No validation** — No train/test split, no overfitting detection, no metrics beyond raw output.
5. **No learning rate** — The weight update step has no tunable learning rate; adjustments are applied directly. This works for this simple dataset but would diverge on more complex data.
6. **No bias parameter** — Relies on a constantly-`1` input feature instead of an explicit trainable bias.
7. **No serialization** — Trained weights are not saved to disk; the model must be retrained every session.
8. **Animation is matplotlib-only** — Requires a GUI backend (TkAgg, Qt5Agg, etc). Won't render in headless environments without X forwarding.

---

## Credits & References

This project is a learning exercise based on tutorials by:

- **Milo Spencer-Harber** — [How to build a simple neural network in 9 lines of Python code](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1)
- **Andrew Trask** — [A Neural Network in 11 lines of Python](https://iamtrask.github.io/2015/07/12/basic-python-network/)
