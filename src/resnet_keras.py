"""
Implementación simple de bloques residuales y un modelo tipo ResNet en Keras
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def identity_block(X, f, filters, block_name="id"):
    """
    Bloque de identidad: no cambia tamaño espacial ni nº de canales del shortcut.

    Argumentos:
    - X: tensor de entrada
    - f: tamaño del kernel para la convolución central
    - filters: lista/tupla de 3 enteros, nº de filtros en cada conv
    """
    F1, F2, F3 = filters

    X_shortcut = X

    # Primer componente
    X = layers.Conv2D(F1, (1, 1), strides=(1, 1), padding="valid", name=f"{block_name}_conv1")(X)
    X = layers.BatchNormalization(axis=3, name=f"{block_name}_bn1")(X)
    X = layers.Activation("relu", name=f"{block_name}_relu1")(X)

    # Segundo componente
    X = layers.Conv2D(F2, (f, f), strides=(1, 1), padding="same", name=f"{block_name}_conv2")(X)
    X = layers.BatchNormalization(axis=3, name=f"{block_name}_bn2")(X)
    X = layers.Activation("relu", name=f"{block_name}_relu2")(X)

    # Tercer componente
    X = layers.Conv2D(F3, (1, 1), strides=(1, 1), padding="valid", name=f"{block_name}_conv3")(X)
    X = layers.BatchNormalization(axis=3, name=f"{block_name}_bn3")(X)

    # Suma con el atajo
    X = layers.Add(name=f"{block_name}_add")([X, X_shortcut])
    X = layers.Activation("relu", name=f"{block_name}_out")(X)

    return X


def convolutional_block(X, f, filters, s=2, block_name="conv"):
    """
    Bloque residual con convolución en el atajo para cambiar tamaño/canales.

    Argumentos:
    - X: tensor de entrada
    - f: tamaño del kernel para la convolución central
    - filters: lista/tupla de 3 enteros, nº de filtros en cada conv
    - s: stride para la primera conv (reduce resolución)
    """
    F1, F2, F3 = filters

    X_shortcut = X

    # Primer componente
    X = layers.Conv2D(F1, (1, 1), strides=(s, s), padding="valid", name=f"{block_name}_conv1")(X)
    X = layers.BatchNormalization(axis=3, name=f"{block_name}_bn1")(X)
    X = layers.Activation("relu", name=f"{block_name}_relu1")(X)

    # Segundo componente
    X = layers.Conv2D(F2, (f, f), strides=(1, 1), padding="same", name=f"{block_name}_conv2")(X)
    X = layers.BatchNormalization(axis=3, name=f"{block_name}_bn2")(X)
    X = layers.Activation("relu", name=f"{block_name}_relu2")(X)

    # Tercer componente
    X = layers.Conv2D(F3, (1, 1), strides=(1, 1), padding="valid", name=f"{block_name}_conv3")(X)
    X = layers.BatchNormalization(axis=3, name=f"{block_name}_bn3")(X)

    # Atajo con conv 1x1 para ajustar dimensiones
    X_shortcut = layers.Conv2D(
        F3, (1, 1), strides=(s, s), padding="valid", name=f"{block_name}_conv_shortcut"
    )(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis=3, name=f"{block_name}_bn_shortcut")(X_shortcut)

    # Suma y activación final
    X = layers.Add(name=f"{block_name}_add")([X, X_shortcut])
    X = layers.Activation("relu", name=f"{block_name}_out")(X)

    return X


def build_small_resnet(input_shape=(64, 64, 3), classes=6):
    """
    Versión pequeña de ResNet, útil para probar rápidamente la arquitectura.
    """
    X_input = layers.Input(input_shape)

    # Etapa inicial
    X = layers.ZeroPadding2D((3, 3))(X_input)
    X = layers.Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(X)
    X = layers.BatchNormalization(axis=3, name="bn_conv1")(X)
    X = layers.Activation("relu")(X)
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Bloques residuales (muy reducidos para hacerlo ligero)
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1, block_name="conv2_block1")
    X = identity_block(X, 3, [64, 64, 256], block_name="conv2_block2")

    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2, block_name="conv3_block1")
    X = identity_block(X, 3, [128, 128, 512], block_name="conv3_block2")

    # Capa final (global average pooling evita errores por tamaño espacial pequeño)
    X = layers.GlobalAveragePooling2D(name="gap")(X)
    X = layers.Dense(classes, activation="softmax", name="fc")(X)

    model = Model(inputs=X_input, outputs=X, name="SmallResNetKeras")
    return model


def build_resnet50(input_shape=(64, 64, 3), classes=6):
    """
    ResNet-50 (arquitectura por etapas) en el mismo estilo del notebook original:
    CONV(7x7,s=2) -> BN -> ReLU -> MAXPOOL(3x3,s=2) ->
    conv2_x (1 conv block + 2 id blocks) ->
    conv3_x (1 conv block + 3 id blocks) ->
    conv4_x (1 conv block + 5 id blocks) ->
    conv5_x (1 conv block + 2 id blocks) ->
    AVGPOOL(2x2) -> FLATTEN -> DENSE(softmax)

    Número de bloques:
    - conv2_x: 1 bloque convolucional + 2 bloques de identidad
    - conv3_x: 1 bloque convolucional + 3 bloques de identidad
    - conv4_x: 1 bloque convolucional + 5 bloques de identidad
    - conv5_x: 1 bloque convolucional + 2 bloques de identidad
    """
    X_input = layers.Input(input_shape)

    # Etapa inicial
    X = layers.ZeroPadding2D((3, 3))(X_input)
    X = layers.Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(X)
    X = layers.BatchNormalization(axis=3, name="bn_conv1")(X)
    X = layers.Activation("relu")(X)
    X = layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # conv2_x
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1, block_name="conv2_block1")
    X = identity_block(X, 3, [64, 64, 256], block_name="conv2_block2")
    X = identity_block(X, 3, [64, 64, 256], block_name="conv2_block3")

    # conv3_x
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2, block_name="conv3_block1")
    X = identity_block(X, 3, [128, 128, 512], block_name="conv3_block2")
    X = identity_block(X, 3, [128, 128, 512], block_name="conv3_block3")
    X = identity_block(X, 3, [128, 128, 512], block_name="conv3_block4")

    # conv4_x
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2, block_name="conv4_block1")
    X = identity_block(X, 3, [256, 256, 1024], block_name="conv4_block2")
    X = identity_block(X, 3, [256, 256, 1024], block_name="conv4_block3")
    X = identity_block(X, 3, [256, 256, 1024], block_name="conv4_block4")
    X = identity_block(X, 3, [256, 256, 1024], block_name="conv4_block5")
    X = identity_block(X, 3, [256, 256, 1024], block_name="conv4_block6")

    # conv5_x
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2, block_name="conv5_block1")
    X = identity_block(X, 3, [512, 512, 2048], block_name="conv5_block2")
    X = identity_block(X, 3, [512, 512, 2048], block_name="conv5_block3")

    # Capa final
    X = layers.AveragePooling2D(pool_size=(2, 2), name="avg_pool")(X)
    X = layers.Flatten(name="flatten")(X)
    
    X = layers.Dropout(0.5, name="dropout_final")(X) 
    
    X = layers.Dense(classes, activation="softmax", name="fc")(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet50")
    return model

    """
    Variante práctica para CIFAR-10 (32x32).

    Mantiene los bloques ResNet-50 (3-4-6-3), pero usa:
    - Un 'stem' más suave (conv 3x3 stride 1, sin maxpool)
    - GlobalAveragePooling2D al final (para evitar el caso 1x1 + pool 2x2)
    """
    X_input = layers.Input(input_shape)

    # Stem CIFAR
    X = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="cifar_conv1")(X_input)
    X = layers.BatchNormalization(axis=3, name="cifar_bn1")(X)
    X = layers.Activation("relu", name="cifar_relu1")(X)

    # conv2_x
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1, block_name="conv2_block1")
    X = identity_block(X, 3, [64, 64, 256], block_name="conv2_block2")
    X = identity_block(X, 3, [64, 64, 256], block_name="conv2_block3")

    # conv3_x
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2, block_name="conv3_block1")
    X = identity_block(X, 3, [128, 128, 512], block_name="conv3_block2")
    X = identity_block(X, 3, [128, 128, 512], block_name="conv3_block3")
    X = identity_block(X, 3, [128, 128, 512], block_name="conv3_block4")

    # conv4_x
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2, block_name="conv4_block1")
    X = identity_block(X, 3, [256, 256, 1024], block_name="conv4_block2")
    X = identity_block(X, 3, [256, 256, 1024], block_name="conv4_block3")
    X = identity_block(X, 3, [256, 256, 1024], block_name="conv4_block4")
    X = identity_block(X, 3, [256, 256, 1024], block_name="conv4_block5")
    X = identity_block(X, 3, [256, 256, 1024], block_name="conv4_block6")

    # conv5_x
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2, block_name="conv5_block1")
    X = identity_block(X, 3, [512, 512, 2048], block_name="conv5_block2")
    X = identity_block(X, 3, [512, 512, 2048], block_name="conv5_block3")

    X = layers.GlobalAveragePooling2D(name="gap")(X)
    X = layers.Dense(classes, activation="softmax", name="fc")(X)

    model = Model(inputs=X_input, outputs=X, name="ResNet50_CIFAR")
    return model


if __name__ == "__main__":
    model = build_resnet50()
    model.summary()

