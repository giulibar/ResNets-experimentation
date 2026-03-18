import json
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import optimizers, layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from resnet_keras import build_resnet50

CONFIG = {
    "dataset_name": "cifar10",
    "target_size": (160, 160),
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "early_stopping_patience": 8,
    "reduce_lr_factor": 0.2,
    "reduce_lr_patience": 5,
    "initial_epoch": 0
}

def _save_run(model, history, run_info: dict, base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_info.get('dataset_name', 'run')}_{timestamp}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config_to_save = {
        "model_name": model.name,
        "total_parameters": model.count_params(),
        **run_info
    }
    
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    if history is not None and hasattr(history, "history"):
        h = {k: [float(val) for val in v] for k, v in history.history.items()}
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(h, f, indent=2)

    # model.save(os.path.join(run_dir, "model_final.keras"))
    return run_dir

def get_cifar10_dataset(batch_size, target_size):
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)

    def build_pipeline(x, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle: ds = ds.shuffle(buffer_size=1024)
        ds = ds.batch(batch_size)
        ds = ds.map(lambda i, j: (tf.image.resize(i, target_size), j), 
                    num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return build_pipeline(x_train, y_train, shuffle=True), build_pipeline(x_val, y_val)

def train_with_cifar(model=None, config=CONFIG, project_path=".", checkpoint_name="resnet_best.keras", extra_callbacks=None):
    checkpoint_dir = os.path.join(project_path, "checkpoints")
    base_runs_dir = os.path.join(project_path, "runs")
    
    train_ds, val_ds = get_cifar10_dataset(
        batch_size=config["batch_size"], 
        target_size=config["target_size"]
    )

    if model is None:
        print("Initializing new model with aggressive augmentation...")
        # --- DATA AUGMENTATION MEJORADO ---
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),       # Un poco más de rotación
            layers.RandomTranslation(0.1, 0.1), # Mueve la imagen para que no memorice la posición
            layers.RandomZoom(0.1),            # Zoom aleatorio para variar escalas
            layers.RandomContrast(0.1),        # Variación de iluminación
        ], name="data_augmentation")

        input_shape = (*config["target_size"], 3)
        inputs = layers.Input(shape=input_shape)
        x = data_augmentation(inputs)
        
        # Construcción de la base
        base_resnet = build_resnet50(input_shape=input_shape, classes=10)
        outputs = base_resnet(x)
        model = tf.keras.Model(inputs, outputs)

    # --- OPTIMIZADOR CON WEIGHT DECAY ---
    # Nota: Asegúrate de que tu CONFIG["weight_decay"] sea al menos 5e-4 o 1e-3 para frenar el overfitting
    optimizer = optimizers.SGD(
        learning_rate=config["learning_rate"], 
        momentum=config["momentum"], 
        weight_decay=config["weight_decay"] 
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_full_path = os.path.join(checkpoint_dir, checkpoint_name)

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_full_path, 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', 
            patience=config["early_stopping_patience"], 
            restore_best_weights=True, 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=config["reduce_lr_factor"], 
            patience=config["reduce_lr_patience"], 
            verbose=1
        )
    ]

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    history = model.fit(
        train_ds,
        epochs=config["epochs"],
        validation_data=val_ds,
        initial_epoch=config["initial_epoch"],
        callbacks=callbacks,
    )

    model.save(os.path.join(project_path, "model_final.keras"))

    run_info = {**config}
    _save_run(model, history, run_info, base_dir=base_runs_dir)

    return history, model
__all__ = ["train_with_cifar"]