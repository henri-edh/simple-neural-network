# Simple Neural Network

A single-layer perceptron in Python — watch synaptic weights converge live as the network learns to predict binary output from 3 inputs.

- **No hidden layers** — the simplest possible neural network.
- **Live training animation** — matplotlib dashboard with weights, predictions, and error curves in real time.
- **Two implementations** — a procedural script (`training_version.py`) for learning the mechanics, and an OOP class (`main.py`) you can import and reuse.

> Built from tutorials by [Milo Spencer-Harber](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1) and [Andrew Trask](https://iamtrask.github.io/2015/07/12/basic-python-network/).

## What does it do?

Given 3 binary inputs, the network learns that **output = Input 1**. If the first input is 1, the output should be 1; otherwise 0. The other two inputs are distractors — the network must discover which one matters.

## Quickstart

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & run — uv handles Python + deps automatically
git clone <repo-url>
cd simple-neural-network
uv run main.py
```

The live training dashboard will open, animate for ~10 seconds, then prompt you to test with your own 3 binary inputs.

## Project layout

```
simple-neural-network/
├── main.py                # OOP NeuralNetwork class (importable, interactive, animated)
├── training_version.py    # Procedural script (educational, one-shot)
├── pyproject.toml          # uv project config & dependencies
├── .python-version         # Pin Python 3.12
├── outputs.txt             # Terminal output of training_version.py
├── README.md
└── architecture.md         # Full architecture deep-dive
```
