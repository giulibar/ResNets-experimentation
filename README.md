## Proyecto: ResNet simple desde cero

Este repositorio contiene una **implementación propia y simplificada de una Red Residual (ResNet)** para clasificación de imágenes, pensada como proyecto de portafolio.

> ⚠️ **Nota importante sobre el honor code**  
> Este proyecto está **inspirado** en la Specialization de *Deep Learning* de Andrew Ng (Coursera), pero **no contiene el notebook original ni las soluciones de los ejercicios**.  
> Todo el código que aparece aquí ha sido reescrito desde cero, con modificaciones y comentarios propios, y usando un flujo de trabajo independiente.

### Objetivos del proyecto

- **Implementar bloques residuales** (skip connections) desde cero.
- Construir una **ResNet pequeña** adecuada para un dataset como CIFAR-10 o similar.
- Entrenar y evaluar el modelo en un dataset de ejemplo.
- Analizar resultados y mostrar el entendimiento de:
  - La idea de aprender funciones residuales \(F(x)\) tal que la salida sea \(y = F(x) + x\).
  - El problema de **degradación en redes profundas** y cómo las skip connections ayudan a mitigarlo.

### Estructura del repositorio

- `README.md`: este documento.
- `requirements.txt`: dependencias del proyecto (TensorFlow, NumPy, Matplotlib).
- `src/`
  - `resnet_keras.py`: implementación de bloques residuales (`identity_block`, `convolutional_block`) y de una ResNet pequeña en Keras.
  - `train_keras.py`: script de entrenamiento configurable:
    - **Modo CIFAR-10** (por defecto): carga el dataset desde Keras y entrena el modelo.
    - **Modo directorio**: entrena con un dataset propio de imágenes organizado en subcarpetas por clase.
- `notebooks/`
  - `resnet_experiments.ipynb`: notebook con experimentos, gráficas y análisis (carga el modelo entrenado y explora sus resultados).

### Instalación

1. Clona este repositorio:

```bash
git clone <TU_URL_DEL_REPO>.git
cd resnet-from-scratch
```

2. (Opcional pero recomendado) Crea y activa un entorno virtual:

```bash
python -m venv venv
.\venv\Scripts\activate  # En Windows
# source venv/bin/activate  # En Linux/Mac
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

### Uso rápido

- **Entrenar usando CIFAR-10 (recomendado como demo reproducible)**:

```bash
python src/train_keras.py --dataset cifar --epochs 10
```

- **Entrenar usando tu propio dataset** (imágenes en subcarpetas dentro de `data/`):

```bash
python src/train_keras.py --dataset directory --data_dir data --epochs 10
```

En ambos casos se guarda el modelo entrenado como `small_resnet_keras.h5` en la raíz del proyecto.

Después, puedes abrir el notebook de experimentos:

```bash
jupyter notebook notebooks/resnet_experiments.ipynb
```

Ahí se carga el modelo guardado y se realizan:

- Evaluaciones en el conjunto de validación/test.
- Visualizaciones de predicciones.
- Comentarios sobre el comportamiento de la red y el papel de las skip connections.

### Créditos

- Inspirado por la **Deep Learning Specialization** de Andrew Ng.  
- Implementación y organización del proyecto: **Giuli**.

