## Project: Simple ResNet from Scratch

This repository contains a **custom, simplified implementation of a Residual Network (ResNet)** for image classification.


### Project Goals

- **Implement residual blocks** (skip connections) from scratch.
- Build a **small ResNet** suitable complex datasets.
- Train and evaluate the model on a sample dataset.
- Analyze results and demonstrate understanding of:
  - The idea of learning residual functions \(F(x)\) such that the output is \(y = F(x) + x\).
  - The **degradation problem in deep networks** and how skip connections help mitigate it.

### Repository Structure

- `README.md`: this document.
- `requirements.txt`: project dependencies (TensorFlow, NumPy, Matplotlib).
- `src/`
  - `resnet_keras.py`: implementation of residual blocks (`identity_block`, `convolutional_block`) and a small ResNet in Keras.
  - `train_keras.py`: configurable training script.
- `notebooks/`
  - `resnet_experiments.ipynb`: notebook with experiments, plots, and analysis (loads the trained model and explores its results).

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

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

- **Train using CIFAR-10 (recommended as a reproducible demo)**:
```bash
python src/train_keras.py --dataset cifar --epochs 10
```

- **Train using your own dataset** (images in subfolders inside `data/`):
```bash
python src/train_keras.py --dataset directory --data_dir data --epochs 10
```

In both cases the trained model is saved as `small_resnet_keras.h5` in the project root.

Afterwards, you can open the experiments notebook:
```bash
jupyter notebook notebooks/resnet_experiments.ipynb
```

There you can find:

- Evaluations on the validation/test set.
- Prediction visualizations.
- Commentary on the network's behavior and the role of skip connections.

### Credits

- Inspired by Andrew Ng's **Deep Learning Specialization**.
- Implementation and project structure: **Giuliano Bardecio**.