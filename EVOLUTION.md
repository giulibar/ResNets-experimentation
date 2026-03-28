# 📈 Log de Experimentos: ResNet-50 en CIFAR-10

Este documento sirve como registro de ingeniería para documentar la evolución del modelo, las hipótesis probadas y los resultados obtenidos en cada iteración del proyecto.

---

## Tabla Comparativa de Resultados

| ID | Description | Epochs | Train Acc | Dev Acc | Test Acc | State |
|:---|:---|:---:|:---:|:---:|:---|:---|
| 01 | **Baseline:** ResNet-50 standard | 20 | 0.9546  | 0.6380 | 0.6378 | Overfitting |
| 02 | **Data Augmentation:** Flip + Rotation | 20 | 0.8065  | 0.7517  | 0.7517 | High bias |  
| 03 | **Train more time:** More epochs + ES | 28 | 0.8986 | 0.7799  | 0.7799 | Overfitting |  
| 04 | **Scale images:** 160x160 + ReduceLR + Dropout | 33 | 0.9850 | 0.8928  | - | Overfitting |  
| 05 | **Checkpointing:** Rescate de pesos óptimos | 40 | 0.9400 | 0.8710 | - | Stable |
| 06 | **Robust Augm:** Translation/Zoom/Contrast + SGD | 80 | 0.9015 | 0.8940 | - | Optimized |

---

## Detalle de Iteraciones

### Experimento #01
* **Arquitectura:** ResNet-50.
* **Configuración Técnica:**

    * **Input Shape:** (64, 64, 3)
    * **Optimizer:** Adam (LR inicial: 0.0001)
    * **Batch Size:** 64
* **Resultados:** Train Acc: 95.46% | Dev Acc: 63.80%
* **Conclusión:** Overfitting severo (gap > 32%). El modelo memoriza el ruido.

---

### Experimento #02
* **Arquitectura:** ResNet-50.
* **Configuración Técnica:**
    * **Data Augmentation:** Flip + Rotation.
    * **Optimizer:** Adam (LR: 0.00005).
* **Resultados:** Train Acc: 80.65% | Dev Acc: 75.17%
* **Conclusión:** La curva se estabilizó y la generalización mejoró notablemente.

---

### Experimento #03
* **Arquitectura:** ResNet-50.
* **Configuración Técnica:**
    * **Epochs:** 28 (Continuación con Early Stopping).
* **Resultados:** Train Acc: 89.86% | Dev Acc: 77.91%
* **Conclusión:** El Early Stopping cortó el entrenamiento al detectar sobreajuste incipiente.

---

### Experimento #04
* **Configuración Técnica:**
    * **Input Shape:** (160, 160, 3)
    * **DropOut:** 0.5 en capa densa final.
    * **Scheduler:** ReduceLROnPlateau.
* **Resultados:** Train Acc: 98.50% | Dev Acc: 89.28%
* **Conclusión:** Problemas de recursos interrumpieron el proceso. Se requiere Model Checkpoint.

---

### Experimento #05
* **Arquitectura:** ResNet-50.
* **Configuración Técnica:**
    * **Input Shape:** (160, 160, 3)
    * **Callbacks:** ModelCheckpoint (monitoreo de val_loss).
    * **Epochs:** 40
* **Resultados:**
    * **Training Accuracy:** 94.00%
    * **Dev Accuracy:** 87.10%
* **Conclusiones:** 1. **Seguridad:** El uso de checkpoints permitió asegurar el mejor estado del modelo ante cortes de ejecución.
    2. **Desempeño:** El modelo es robusto, pero el gap de ~7% indica que aún hay espacio para regularización.

---

### Experimento #06
* **Arquitectura:** ResNet-50.
* **Configuración Técnica:**
    * **Optimizer:** SGD (LR inicial: 0.0020) + Weight Decay.
    * **Data Augmentation Pro:** * RandomTranslation (0.1, 0.1)
        * RandomZoom (0.1)
        * RandomContrast (0.1)
    * **Epochs:** 80 (Continuación desde pesos del Exp #05).
* **Resultados (Métrica a Época 35):**
    * **Training Accuracy:** 90.15%
    * **Dev Accuracy:** 89.40%
    * **Loss:** 0.2812 | **Val Loss:** 0.3319
* **Conclusiones:** 1. **Generalización Casi Perfecta:** El gap se redujo a **0.75%**, indicando que las nuevas capas de aumento impiden que el modelo memorice posiciones o iluminación fijas.
    2. **Estabilidad:** El uso de SGD en etapas avanzadas de entrenamiento favorece una convergencia más fina comparado con Adam.
    3. **Próximos pasos:** Evaluar el Test Acc final una vez completadas las 80 épocas.