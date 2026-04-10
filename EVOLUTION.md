# 📈 Experiment Log: ResNet-50 on CIFAR-10

This document serves as an engineering log to document the model's evolution, tested hypotheses, and results obtained in each project iteration.

> **Note:** ResNet-50 is an excessively large and complex architecture for the CIFAR-10 classification problem. However, the challenge was precisely to explore how to train this model without overfitting and to achieve the best possible performance across successive experiments — all while deepening my understanding of model training through hands-on practice as part my Deep Learning specialization.

---

## Results Summary

| ID | Description | Epochs | Train Acc | Dev Acc | Test Acc | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| 01 | **Baseline:** ResNet-50 standard | 20 | 0.9546 | 0.6380 | 0.6378 | Overfitting |
| 02 | **Data Augmentation:** Flip + Rotation | 20 | 0.8065 | 0.7517 | 0.7517 | High bias |
| 03 | **Train more time:** More epochs + early stopping | 28 | 0.8986 | 0.7799 | 0.7799 | Overfitting |
| 04 | **Scale images, variable LR:** Images 160×160 + ReduceLROnPlateau + Dropout | 33 | 0.9850 | 0.8928 | - | Overfitting |

---

## Experiment Detail

---

### Experiment #01

#### Configuration
| Parameter | Value |
|:---|:---|
| Architecture | ResNet-50 |
| Input Shape | (64, 64, 3) |
| Optimizer | Adam (LR: 0.0001) |
| Batch Size | 64 |
| Normalization | Pixels rescaled to [0, 1] |
| Epochs | 20 |

#### Results
| Metric | Value |
|:---|:---|
| Training Accuracy | 95.46% |
| Dev Accuracy | 63.80% |
| Test Accuracy | 63.78% |

![trainingphoto](/images/1exp.png)

#### Conclusions

1. **Overfitting:** A gap of over 32% is observed between training and test accuracy. The model is memorizing noise from the training set instead of generalizing features.
2. **Validation Instability:** Despite using a conservative LR of 0.0001, the `val_accuracy` curves show instability with large jumps between epochs.
3. **Summary:** The model is not generalizing correctly and its behavior is poor.

---

### Experiment #02

#### Implementations
- Reduced LR from 0.0001 to 0.00005
- Applied data augmentation with random rotation and zoom to prevent overfitting to training data

#### Configuration
| Parameter | Value |
|:---|:---|
| Architecture | ResNet-50 |
| Input Shape | (64, 64, 3) |
| Optimizer | Adam (LR: 0.00005) |
| Batch Size | 64 |
| Normalization | Pixels rescaled to [0, 1] |
| Epochs | 20 |

#### Results
| Metric | Value |
|:---|:---|
| Training Accuracy | 80.65% |
| Dev Accuracy | 75.17% |
| Test Accuracy | 75.17% |

![trainingphoto](/images/2exp.png)

#### Conclusions

1. **Reduced Overfitting:** Dev accuracy improved and the gap with train accuracy is now 5%, suggesting the model is generalizing much better.
2. **Validation Stability:** The LR reduction took effect and the accuracy curve behaves more stably without abrupt jumps.
3. **Summary:** The model generalized much better on the test set. Validation accuracy is still climbing, indicating room for improvement.

---

### Experiment #03

#### Implementations
- Continued training for more epochs
- Implemented early stopping to stop at the point of best model performance

#### Configuration
| Parameter | Value |
|:---|:---|
| Architecture | ResNet-50 |
| Input Shape | (64, 64, 3) |
| Optimizer | Adam (LR: 0.00005) |
| Batch Size | 64 |
| Normalization | Pixels rescaled to [0, 1] |
| Epochs | 28 |

#### Results
| Metric | Value |
|:---|:---|
| Training Accuracy | 89.86% |
| Dev Accuracy | 77.91% |
| Test Accuracy | 77.99% |

![trainingphoto](/images/3exp.png)

Training was resumed from the previous 20-epoch checkpoint to avoid retraining from scratch, and continued for 20 additional epochs with early stopping (patience=7, monitoring `val_loss`) to give the model sufficient time to improve before cutting training.

#### Conclusions

1. **Overfitting:** While the model slightly improved on dev and test, training accuracy was rising much faster than dev accuracy, indicating the model is overfitting again.
2. **Summary:** The model overfit significantly and early stopping cut the training short. Regularization or additional data would benefit performance.

---
### Experiment #04

#### Implementations
- Images rescaled to 160×160
- ReduceLROnPlateau for dynamic LR scheduling
- Dropout (0.5) on final dense layer

#### Configuration
| Parameter | Value |
|:---|:---|
| Architecture | ResNet-50 |
| Input Shape | (160, 160, 3) |
| Optimizer | Adam (LR: 1.0000e-04) |
| Batch Size | 64 |
| Normalization | Pixels rescaled to [0, 1] |
| Dropout | 0.5 on final dense layer |
| Epochs | 33 |

#### Results
| Metric | Value |
|:---|:---|
| Training Accuracy | 98.50% |
| Dev Accuracy | 89.28% |
| Test Accuracy | - |

![trainingphoto](/images/4exp.png)

#### Conclusions

1. **Overfitting:** The model is still overfitting.
2. **Summary:** Resource limitations caused a loss of progress. Model checkpoints will be added to avoid losing progress in case of error.

---

### Experiment #05

#### Implementations
- Weight decay regularization
- Switched optimizer from Adam to SGD
- Added model checkpoints to preserve training progress

#### Configuration
| Parameter | Value |
|:---|:---|
| Architecture | ResNet-50 |
| Input Shape | (160, 160, 3) |
| Optimizer | SGD |
| Normalization | Pixels rescaled to [0, 1] |
| Dropout | 0.5 on final dense layer |

#### Results

| Metric | Value |
|:---|:---|
| Training Accuracy | 94.00% |
| Dev Accuracy | 87.10% |
| Test Accuracy | - |

> Metrics reported at epoch 40.

![trainingphoto](/images/5exp.png)

#### Conclusions

1. **Improved Generalization:** Switching to SGD with weight decay reduced the train/dev gap compared to Experiment #04, showing better regularization.
2. **Summary:** The model shows a healthier training dynamic. Continuing training with additional data augmentation may further close the gap and improve test performance.

---

### Experiment #06

#### Implementations
- Continued training from Experiment #05
- Added data augmentation layers:
  - `RandomTranslation(0.1, 0.1)` — shifts the image so the model doesn't memorize position
  - `RandomZoom(0.1)` — random zoom to vary scales
  - `RandomContrast(0.1)` — lighting variation

#### Configuration
| Parameter | Value |
|:---|:---|
| Architecture | ResNet-50 |
| Input Shape | (160, 160, 3) |
| Optimizer | SGD |
| Normalization | Pixels rescaled to [0, 1] |
| Dropout | 0.5 on final dense layer |
| Epochs | 35 |

#### Results
| Metric | Value |
|:---|:---|
| Training Accuracy | 90.15% |
| Dev Accuracy | 87.17% |
| Dev Loss | 0.4019 |
| Test Accuracy | **89.40%** |
| Test Loss | 0.3550 |
| Learning Rate | 0.0020 |

![trainingphoto](/images/6exp.png)

#### Conclusions

1. **Best generalization so far:** The train/dev gap narrowed to ~3%, the healthiest margin across all experiments.
2. **Strong test performance:** 89.40% test accuracy is the best test result achieved, confirming that the combination of SGD, weight decay, and data augmentation generalizes well to unseen data.
3. **Summary:** Data augmentation effectively reduced overfitting. The model is learning meaningful features rather than memorizing training examples.