## Project: Simple ResNet from Scratch

This repository contains a **custom, simplified implementation of a Residual Network (ResNet)** for image classification on CIFAR-10.

---

### Project Goals

- **Implement residual blocks** (skip connections) from scratch.
- Build a **small ResNet** suitable for CIFAR-10.
- Train and evaluate the model, exploring:
  - The idea of learning residual functions $F(x)$ such that the output is $y = F(x) + x$.
  - The **degradation problem in deep networks** and how skip connections help mitigate it.

---

### Repository Structure


For a detailed log of experiments, results, and observations, see the [EVOLUTION](./EVOLUTION.md) document.


- `README.md`: this document.
- `requirements.txt`: project dependencies (TensorFlow, NumPy, Matplotlib).
- `src/`
  - `resnet_keras.py`: implementation of residual blocks (`identity_block`, `convolutional_block`) and `build_resnet50`, a small ResNet model in Keras.
  - `train_keras.py`: training script for CIFAR-10 with data augmentation, SGD with weight decay, callbacks, and run logging.
- `checkpoints/`: best model weights saved during training (`resnet_best.keras`).
- `runs/`: per-run directories with `config.json` and `history.json` for experiment tracking.
- `notebooks/`
  - `resnet_experiments.ipynb`: notebook with experiments, plots, and analysis.

---

### Training Configuration

| Parameter | Default |
|---|---|
| Input size | 160 × 160 |
| Batch size | 64 |
| Epochs | 20 |
| Optimizer | SGD |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Early stopping patience | 8 |
| LR reduction factor | 0.2 |
| LR reduction patience | 5 |

**Data augmentation:** random horizontal flip, rotation (±15%), translation (±10%), zoom (±10%), and contrast (±10%).

---

### Installation

1. Clone this repository:
```bash
git clone <YOUR_REPO_URL>.git
cd resnet-from-scratch
```

2. (Optional but recommended) Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

### Quick Start

Train the model on CIFAR-10:
```bash
python src/train_keras.py
```

This will:
- Download and preprocess CIFAR-10 automatically.
- Apply data augmentation during training.
- Save the best checkpoint to `checkpoints/resnet_best.keras`.
- Save the final model to `model_final.keras`.
- Log the config and training history to a timestamped folder under `runs/`.

Then, open the experiments notebook:
```bash
jupyter notebook notebooks/resnet_experiments.ipynb
```

The notebook covers:

- Evaluation on the test set.
- Prediction visualizations.
- Discussion of the network's behavior and the role of skip connections.

---

### Credits

- Inspired by Andrew Ng's **Deep Learning Specialization**.
- Implementation and project structure: **Giuli**.