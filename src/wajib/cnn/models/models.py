import numpy as np
from typing import Dict, List, Tuple, Optional
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score
import json
import os
from itertools import product

from src.wajib.cnn.layers.layers import (
    AveragePooling2D,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    LocallyConnected2D,
    MaxPooling2D,
)

def CNN(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    num_conv_layers: int,
    num_filters: List[int],
    filter_sizes: List[int],
    pooling_type: str,
    use_locally_connected: bool = False,
) -> "keras.Model":
    if num_conv_layers < 1:
        raise ValueError("num_conv_layers must be at least 1.")
    if pooling_type not in ("max", "average"):
        raise ValueError("pooling_type must be 'max' or 'average'.")

    filters = _expand_config_list(num_filters, num_conv_layers)
    kernels = _expand_config_list(filter_sizes, num_conv_layers)

    model = keras.Sequential(name=_model_name(num_conv_layers, filters, kernels, pooling_type, use_locally_connected))
    model.add(layers.Input(shape=input_shape))

    for idx in range(num_conv_layers):
        if use_locally_connected:
            locally_connected = getattr(layers, "LocallyConnected2D", None)
            if locally_connected is None:
                raise ImportError("tf.keras.layers.LocallyConnected2D is not available in this TensorFlow/Keras version.")
            model.add(locally_connected(filters[idx], kernels[idx], activation="relu"))
        else:
            model.add(layers.Conv2D(filters[idx], kernels[idx], padding="same", activation="relu"))

        if pooling_type == "max":
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def __train__(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    save_path: Optional[str] = None,
) -> dict:
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    if save_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                save_path,
                monitor="val_loss",
                save_best_only=True,
            )
        )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history.history


def __evaluate__keras(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }


def __evaluate__scratch(
    scratch_model_forward,   
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    y_pred = []
    for sample in X_test:
        pred = scratch_model_forward(sample)
        y_pred.append(int(np.argmax(pred)))

    return {"macro_f1": float(f1_score(y_test, y_pred, average="macro"))}

def __run__(
    X_train, y_train, X_val, y_val, X_test, y_test,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    weights_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
) -> List[Dict]:
    os.makedirs(weights_dir, exist_ok=True)
    results = []

    conv_layer_options = [2]
    filter_options = [[32, 64]]
    filter_size_options = [[3, 3]]
    pooling_options = ["max"]

    for num_conv_layers, num_filters, filter_sizes, pooling_type in product(
        conv_layer_options,
        filter_options,
        filter_size_options,
        pooling_options,
    ):
        filters = _expand_config_list(num_filters, num_conv_layers)
        kernels = _expand_config_list(filter_sizes, num_conv_layers)
        name = _model_name(num_conv_layers, filters, kernels, pooling_type, False)
        save_path = os.path.join(weights_dir, f"{name}.keras")

        model = CNN(
            input_shape=input_shape,
            num_classes=num_classes,
            num_conv_layers=num_conv_layers,
            num_filters=filters,
            filter_sizes=kernels,
            pooling_type=pooling_type,
        )
        history = __train__(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_path,
        )
        metrics = __evaluate__keras(model, X_test, y_test)
        result = {
            "name": name,
            "weights_path": save_path,
            "config": {
                "num_conv_layers": num_conv_layers,
                "num_filters": filters,
                "filter_sizes": kernels,
                "pooling_type": pooling_type,
            },
            "history": history,
            "metrics": metrics,
            "params": int(model.count_params()),
        }
        results.append(result)

        with open(os.path.join(weights_dir, "cnn_experiment_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def _expand_config_list(values: List[int], length: int) -> List[int]:
    if len(values) >= length:
        return list(values[:length])
    if not values:
        raise ValueError("Configuration list cannot be empty.")
    return list(values) + [values[-1]] * (length - len(values))


def _model_name(
    num_conv_layers: int,
    num_filters: List[int],
    filter_sizes: List[int],
    pooling_type: str,
    use_locally_connected: bool,
) -> str:
    layer_type = "local" if use_locally_connected else "conv"
    filters = "-".join(str(v) for v in num_filters)
    kernels = "-".join(str(v) for v in filter_sizes)
    return f"cnn_{layer_type}_L{num_conv_layers}_F{filters}_K{kernels}_P{pooling_type}"


class ScratchCNN:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.forward(sample) for sample in X], axis=0)

    def count_params(self) -> int:
        total = 0
        for layer in self.layers:
            for attr in ("kernel", "bias", "W", "b"):
                value = getattr(layer, attr, None)
                if value is not None:
                    total += int(np.prod(value.shape))
        return total


def scratch_from_keras(keras_model) -> ScratchCNN:
    scratch_layers = []

    for keras_layer in keras_model.layers:
        class_name = keras_layer.__class__.__name__

        if class_name == "Conv2D":
            layer = Conv2D()
            layer.__loadWeights__(keras_layer)
        elif class_name == "LocallyConnected2D":
            layer = LocallyConnected2D()
            layer.__loadWeights__(keras_layer)
        elif class_name == "MaxPooling2D":
            layer = MaxPooling2D(
                pool_size=tuple(keras_layer.pool_size),
                strides=tuple(keras_layer.strides),
            )
        elif class_name == "AveragePooling2D":
            layer = AveragePooling2D(
                pool_size=tuple(keras_layer.pool_size),
                strides=tuple(keras_layer.strides),
            )
        elif class_name == "GlobalMaxPooling2D":
            layer = GlobalMaxPooling2D()
        elif class_name == "GlobalAveragePooling2D":
            layer = GlobalAveragePooling2D()
        elif class_name == "Flatten":
            layer = Flatten()
        elif class_name == "Dense":
            layer = Dense()
            layer.__loadWeights__(keras_layer)
        elif class_name in ("InputLayer", "Dropout", "BatchNormalization"):
            continue
        else:
            raise ValueError(f"Unsupported Keras layer for scratch conversion: {class_name}")

        scratch_layers.append(layer)

    return ScratchCNN(scratch_layers)
