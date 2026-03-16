import json
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from resnet_keras import build_resnet50


def _save_run(model, history, run_info: dict, base_dir: str = "runs") -> str:
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_info.get('dataset', 'run')}_{timestamp}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)


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
        "total_parameters": model.count_params(), 
        "extra": run_info.get("extra", {}),
    }
    
    config["architecture_json"] = json.loads(model.to_json())

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if history is not None and hasattr(history, "history"):
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)

    print(f"Metadata y métricas guardadas en: {run_dir}")
    return run_dir


def train_with_cifar(epochs: int = 5, batch_size: int = 64, learning_rate: float = 1e-4, model=None, initial_epoch=0):
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    
    x_train = tf.image.resize(x_train, (160, 160))
    x_val = tf.image.resize(x_val, (160, 160))

    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)


    if model is None:
        print("Creating new model from scratch...")

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ], name="data_augmentation")


        inputs = layers.Input(shape=(160, 160, 3))
        x = data_augmentation(inputs)
        
        base_resnet = build_resnet50(input_shape=(160, 160, 3), classes=num_classes)
        outputs = base_resnet(x)
        model = tf.keras.Model(inputs, outputs)
    else:
        print("Resuming training with loaded model...")


    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor='val_loss',      
        patience=7,              
        restore_best_weights=True,
        mode='min',              
        verbose=1           
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,         
        patience=3,         
        min_lr=1e-7,
        verbose=1
    )


    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch, 
        validation_data=(x_val, y_val),
        callbacks=[early_stop, reduce_lr],
    )

    run_info = {
        "dataset": "cifar10",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "extra": {"resize": 160, "dropout": 0.5}
    }
    _save_run(model, history, run_info) 

    model.save("resnet_model.h5")

    return history


__all__ = ["train_with_cifar"]