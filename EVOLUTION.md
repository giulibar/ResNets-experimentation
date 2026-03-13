# 📈 Log de Experimentos: ResNet-50 en CIFAR-10

Este documento sirve como registro de ingeniería para documentar la evolución del modelo, las hipótesis probadas y los resultados obtenidos en cada iteración del proyecto.

---

## 📊 Tabla Comparativa de Resultados

| ID | Description | Epochs | Train Acc | Dev Acc | Test Acc | State |
|:---|:---|:---:|:---:|:---:|:---|:---|
| 01 | **Baseline:** ResNet-50 standard | 20 | 0.9546  | 0.6380 | 0.6378 | Overfitting |
| 02 | **Data Augmentation:** Flip + Rotation | - | - | - | - | - |  

---

## 🔬 Detalle de Iteraciones

### 🏷️ Experimento #01: Modelo Base (Baseline)
* **Fecha:** 13 de Marzo, 2026
* **Arquitectura:** ResNet-50.
* **Configuración Técnica:**
    * **Input Shape:** (32, 32, 3)
    * **Optimizer:** Adam (LR inicial: 0.0001)
    * **Batch Size:** 64
    * **Normalización:** Píxeles reescalados a [0, 1].

#### 📈 Resultados
* **Training Accuracy:** 95.46%
* **Dev Accuracy:** 63.80%
* **Test Accuracy:** 63.78%



#### 📝 Conclusiones y Análisis Técnico
1.  **Overfitting Crítico:** Se observa una brecha (gap) de más del 21% entre el entrenamiento y el test. El modelo ha comenzado a memorizar el ruido del set de entrenamiento en lugar de generalizar características.
2.  **Inestabilidad en Validación:** A pesar de usar un LR conservador de 0.0001, las curvas de *val_accuracy* muestran varianza. Esto sugiere que el modelo se beneficia de una estrategia de regularización más fuerte (Data Augmentation) o un batch size mayor.
3.  **Resolución y Arquitectura:** Se implementó un reescalado preventivo de 32x32 a 64x64 píxeles. Sin esta transformación, la profundidad de la ResNet-50 colapsaría los mapas de características a 1x1, impidiendo el aprendizaje de patrones espaciales complejos en clases visualmente similares.

---

### 🏷️ Experimento #02: Regularización mediante Data Augmentation
* **Hipótesis:** Implementar transformaciones aleatorias en las imágenes de entrada reducirá la capacidad del modelo de memorizar píxeles exactos, forzándolo a aprender patrones espaciales más robustos.
* **Cambios a realizar:**
    * Añadir capa de `RandomFlip("horizontal")`.
    * Añadir capa de `RandomRotation(0.1)`.
* **Estado:** ⏳ Por ejecutar.

---