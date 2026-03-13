"""
Script de entrenamiento para SmallResNetKeras con dos funciones principales:

- `train_with_cifar(...)`: entrena usando CIFAR-10 (descargado automáticamente).
- `train_with_directory(...)`: entrena usando un directorio con imágenes propias.

Cada entrenamiento guarda:
- Un archivo `small_resnet_keras.h5` con el último modelo (para el notebook).
- Una carpeta única dentro de `runs/` con:
  - `model.h5`  -> pesos + arquitectura de ese entrenamiento.
  - `config.json` -> hiperparámetros, tipo de dataset, info de la arquitectura.
  - `history.json` -> curvas de loss/accuracy de entrenamiento y validación.
"""

import json
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from resnet_keras import build_resnet50, build_resnet50_cifar, build_resnet50_cifar


def _save_run(model, history, run_info: dict, base_dir: str = "runs") -> str:
    """
    Guarda el modelo, la configuración y el histórico de entrenamiento
    en una carpeta única dentro de `base_dir` y devuelve la ruta.
    """
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_info.get('dataset', 'run')}_{timestamp}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Modelo (arquitectura + pesos)
    model_path = os.path.join(run_dir, "model.h5")
    model.save(model_path)

    # Configuración e hiperparámetros
    # input_shape puede ser una tupla o un objeto similar a tupla -> lo convertimos a lista
    input_shape = model.input_shape
    try:
        input_shape = list(input_shape)
    except TypeError:
        input_shape = str(input_shape)

    config = {
        "model_name": model.name,
        "input_shape": input_shape,
        "dataset": run_info.get("dataset"),
        "epochs": run_info.get("epochs"),
        "batch_size": run_info.get("batch_size"),
        "learning_rate": run_info.get("learning_rate"),
        "extra": run_info.get("extra", {}),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Histórico de entrenamiento (loss, accuracy, etc.)
    if history is not None and hasattr(history, "history"):
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)

    print(f"Entrenamiento guardado en: {run_dir}")
    return run_dir


def train_with_cifar(epochs: int = 5, batch_size: int = 64, learning_rate: float = 1e-4):
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

    # Normalización y reescalado a 64x64 (como el notebook original)
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    
    # Reescalar de 32x32 a 64x64
    x_train = tf.image.resize(x_train, (64, 64))
    x_val = tf.image.resize(x_val, (64, 64))

    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)

    model = build_resnet50(input_shape=(64, 64, 3), classes=num_classes)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
    )

    # Guardado genérico para el notebook (último modelo)
    model.save("small_resnet_keras.h5")
    print("Modelo (CIFAR-10) guardado en small_resnet_keras.h5")

    # Guardar carpeta de experimento con pesos + config + history
    _save_run(
        model,
        history,
        run_info={
            "dataset": "cifar10",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "extra": {"num_classes": num_classes},
        },
    )

    return history


def train_with_directory(
    data_dir: str = "data",
    img_size=(64, 64),
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1.5e-4,
):

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )

    num_classes = train_gen.num_classes

    # Para datasets propios (normalmente 64x64 como en el notebook original),
    # usamos la arquitectura estilo notebook.
    model = build_resnet50(input_shape=img_size + (3,), classes=num_classes)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
    )

    # Guardado genérico para el notebook (último modelo)
    model.save("small_resnet_keras.h5")
    print("Modelo (directorio) guardado en small_resnet_keras.h5")

    # Guardar carpeta de experimento con pesos + config + history
    _save_run(
        model,
        history,
        run_info={
            "dataset": "directory",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "extra": {
                "num_classes": num_classes,
                "data_dir": data_dir,
                "img_size": img_size,
            },
        },
    )

    return history


__all__ = ["train_with_cifar", "train_with_directory"]