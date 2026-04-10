# 📈 Experiment Log: ResNet-50 on CIFAR-10

This document serves as an engineering log to track the model's evolution, the hypotheses tested, and the results obtained in each iteration of the project.


The `.keras` file with the trained model weights can be found at: https://drive.google.com/file/d/1_1GQkDA4Zoshx7XEG795XbypJofAEwDQ/view?usp=sharing

---

## Comparative Results Table

| ID | Description | Epochs | Train Acc | Dev Acc | Test Acc | State |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| 01 | **Baseline:** ResNet-50 standard | 20 | 0.9546 | 0.6380 | 0.6378 | Overfitting |
| 02 | **Data Augmentation:** Flip + Rotation | 20 | 0.8065 | 0.7517 | 0.7517 | High bias |
| 03 | **Train more time:** More epochs + early stopping | 28 | 0.8986 | 0.7799 | 0.7799 | Overfitting |
| 04 | **Scale images, variable LR:** 160x160 + ReduceLROnPlateau + Dropout | 33 | 0.9850 | 0.8928 | - | Overfitting |
| 05 | **Optimizer swap:** SGD + weight decay + more epochs | 40 | 0.9400 | 0.8710 | 0.8710 | Overfitting |
| 06 | **Extended training + data augmentation:** Translation + Zoom + Contrast | 35 | 0.9215 | 0.8997 | 0.8940 | Overfitting |

---

## Iteration Details

To train the model on CIFAR-10 data using the ResNet-50 architecture (which is very deep), an upscaling from 32×32 to 64×64 pixels was required. Without this transformation, the depth of ResNet-50 collapsed the feature maps to 1×1, preventing the model from learning complex spatial patterns in visually similar classes.

---

### Experiment #01
* **Architecture:** ResNet-50
* **Technical Configuration:**
    * **Input Shape:** (64, 64, 3)
    * **Optimizer:** Adam (Initial LR: 0.0001)
    * **Batch Size:** 64
    * **Normalization:** Pixels rescaled to [0, 1]
    * **Epochs:** 20

#### Results
* **Training Accuracy:** 95.46%
* **Dev Accuracy:** 63.80%
* **Test Accuracy:** 63.78%

![trainingphoto](/images/1exp.png)

#### Conclusions

1. **Overfitting:** A gap of over 32% is observed between training and test accuracy. The model is memorizing noise from the training set rather than generalizing features.
2. **Validation Instability:** Despite using a conservative LR of 0.0001, the *val_accuracy* curves show instability with large jumps between epochs.
3. **Conclusion:** The model is not generalizing correctly and its performance is poor.

---

### Experiment #02

Based on the findings from Experiment #01, the learning rate was reduced from 0.0001 to 0.00005 to smooth the validation curve. Data augmentation with random rotation and zoom was also introduced to prevent the model from overfitting to training samples.

* **Architecture:** ResNet-50
* **Technical Configuration:**
    * **Input Shape:** (64, 64, 3)
    * **Optimizer:** Adam (Initial LR: 0.00005)
    * **Batch Size:** 64
    * **Normalization:** Pixels rescaled to [0, 1]
    * **Epochs:** 20

#### Results
* **Training Accuracy:** 80.65%
* **Dev Accuracy:** 75.17%
* **Test Accuracy:** 75.17%

![trainingphoto](/images/2exp.png)

#### Conclusions

1. **Reduced Overfitting:** Dev accuracy improved significantly and the gap with train accuracy dropped to ~5%, suggesting the model is generalizing much better.
2. **Improved Stability:** The LR reduction had a clear effect — the accuracy curve is smoother with fewer abrupt jumps.
3. **Conclusion:** The model generalized much better on the test set. Since validation accuracy is still trending upward, there is clear room for improvement.

---

### Experiment #03

Since validation accuracy was still rising at the end of Experiment #02, training was resumed from the 20-epoch checkpoint (avoiding full retraining) and extended for 20 additional epochs. Early stopping with a patience of 7 was applied, monitoring *val_loss* to allow the model enough time to improve before halting.

* **Architecture:** ResNet-50
* **Technical Configuration:**
    * **Input Shape:** (64, 64, 3)
    * **Optimizer:** Adam (Initial LR: 0.00005)
    * **Batch Size:** 64
    * **Normalization:** Pixels rescaled to [0, 1]
    * **Epochs:** 28

#### Results
* **Training Accuracy:** 89.86%
* **Dev Accuracy:** 77.91%
* **Test Accuracy:** 77.99%

![trainingphoto](/images/3exp.png)

#### Conclusions

1. **Overfitting:** Although dev and test performance improved slightly, training accuracy rose much faster than dev accuracy, indicating the model is overfitting again.
2. **Conclusion:** The model overfit significantly and early stopping cut training short. Stronger regularization and/or more data diversity would benefit the next iteration.

---
### Experiment #04

### Experiment #04

To address the recurring overfitting, three changes were applied simultaneously: images were upscaled to 160×160 to give the model richer spatial information, `ReduceLROnPlateau` was introduced to automatically lower the learning rate when validation stalls, and a Dropout layer (0.5) was added to the final dense layer for regularization. Model checkpointing was also enabled to avoid losing progress in case of resource errors.

* **Architecture:** ResNet-50
* **Technical Configuration:**
    * **Input Shape:** (160, 160, 3)
    * **Optimizer:** Adam (Initial LR: 1e-04)
    * **Batch Size:** 64
    * **Normalization:** Pixels rescaled to [0, 1]
    * **Dropout:** 0.5 on final dense layer
    * **LR Scheduler:** ReduceLROnPlateau
    * **Epochs:** 33

#### Results
* **Training Accuracy:** 98.50%
* **Dev Accuracy:** 89.28%
* **Test Accuracy:** —

![trainingphoto](/images/4exp.png)

#### Conclusions

1. **Overfitting:** Despite the regularization improvements, a significant gap between training and dev accuracy remains. Test accuracy was not recorded due to resource limitations.
2. **Conclusion:** The higher resolution input clearly boosted dev accuracy to 89.28%, but the model is still overfitting. The Adam optimizer may be contributing to sharp minima that hurt generalization — switching to SGD with weight decay and training for more epochs with checkpointing were identified as the next steps.

---

### Experiment #05

Following the overfitting observed in Experiment #04, the optimizer was switched from Adam to SGD, which tends to find flatter minima and generalize better. Weight decay was added as an additional regularization mechanism. Training was extended to 40 epochs with model checkpointing enabled to prevent losing progress.

* **Architecture:** ResNet-50
* **Technical Configuration:**
    * **Input Shape:** (160, 160, 3)
    * **Optimizer:** SGD + weight decay
    * **Batch Size:** 64
    * **Normalization:** Pixels rescaled to [0, 1]
    * **Dropout:** 0.5 on final dense layer
    * **LR Scheduler:** ReduceLROnPlateau
    * **Model Checkpointing:** Enabled
    * **Epochs:** 40

#### Results
* **Training Accuracy:** 94.00%
* **Dev Accuracy:** 87.10%
* **Test Accuracy:** 87.10%

#### Results

#### Conclusions

1. **Overfitting:** The train/dev gap narrowed compared to Experiment #04 (from ~9% to ~7%), indicating that SGD and weight decay had a positive regularization effect.
2. **Conclusion:** Switching to SGD improved generalization noticeably. However, the model is still overfitting. Expanding data augmentation with translation, zoom, and contrast variation was identified as the next step to further reduce the gap.

---

### Experiment #06

Building on Experiment #05, the data augmentation pipeline was significantly expanded to expose the model to more varied inputs and reduce its ability to memorize spatial patterns. Three new augmentation layers were added: random translation, random zoom, and random contrast. Training ran until early stopping triggered at epoch 35 due to overfitting.

* **Architecture:** ResNet-50
* **Technical Configuration:**
    * **Input Shape:** (160, 160, 3)
    * **Optimizer:** SGD + weight decay
    * **Batch Size:** 64
    * **Normalization:** Pixels rescaled to [0, 1]
    * **Dropout:** 0.5 on final dense layer
    * **LR Scheduler:** ReduceLROnPlateau
    * **Model Checkpointing:** Enabled
    * **Data Augmentation:**
        * `RandomTranslation(0.1, 0.1)` — shifts the image to prevent memorization of position
        * `RandomZoom(0.1)` — random zoom to vary scales
        * `RandomContrast(0.1)` — lighting variation
    * **Epochs:** 35 (early stopping)

#### Results
* **Training Accuracy:** 92.15%
* **Dev Accuracy:** 89.97%
* **Test Accuracy:** 89.40%

![trainingphoto](/images/6exp.png)

#### Conclusions

1. **Best generalization so far:** The expanded augmentation successfully reduced the train/dev gap to under 3%, and test accuracy reached 89.40% — the best result across all experiments.
2. **Overfitting:** The model still shows signs of overfitting, with early stopping triggering at epoch 35. Further regularization or architectural changes may be needed to push beyond 90%.
3. **Conclusion:** The combination of SGD, weight decay, dropout, and richer data augmentation produced the strongest overall performance. Reaching near 90% test accuracy on CIFAR-10 with ResNet-50 represents a solid result, though the persistent overfitting suggests there is still headroom to explore.
