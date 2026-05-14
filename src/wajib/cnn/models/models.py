import numpy as np
import json
import os
from itertools import product
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score

from ..layers.layers import (
    AveragePooling2D, Conv2D, Dense, Flatten,
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    LocallyConnected2D, MaxPooling2D,
)


def CNN(
        input_shape, 
        num_classes, 
        num_conv_layers, 
        num_filters,
        filter_sizes,
        pooling_type,
        use_locally_connected=False
    ):
    if num_conv_layers < 1:
        raise ValueError("num_conv_layers must be at least 1.")
    if pooling_type not in ('max', 'average'):
        raise ValueError("pooling_type must be 'max' or 'average'.")

    filters = expandConfigList(num_filters, num_conv_layers)
    kernels = expandConfigList(filter_sizes, num_conv_layers)

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for idx in range(num_conv_layers):
        model.add(layers.Conv2D(filters[idx], kernels[idx], padding='same', activation='relu'))
        model.add(layers.BatchNormalization()) 

        if pooling_type == 'max':
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        
        model.add(layers.Dropout(0.2)) 

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu')) 
    model.add(layers.Dropout(0.5)) 
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def trainModel(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=32,
        save_path=None,
):
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    if save_path:
        callbacks.append(keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    return history.history


def evaluateKeras(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
    }


def evaluateScratch(scratch_model_forward, X_test, y_test):
    y_pred = [int(np.argmax(scratch_model_forward(sample))) for sample in X_test]
    return {'macro_f1': float(f1_score(y_test, y_pred, average='macro'))}


def runExperiments(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        input_shape,
        num_classes,
        weights_dir,
        epochs=10,
        batch_size=32,
):
    os.makedirs(weights_dir, exist_ok=True)
    results = []

    variations = product(
        [2, 3],                      # num_conv_layers
        [[32, 64], [64, 128]],       # num_filters
        [[3, 3], [5, 5]],            # filter_sizes
        ['max', 'average']           # pooling_type
    )

    for i, (num_conv_layers, num_filters, filter_sizes, pooling_type) in enumerate(variations):
        filters = expandConfigList(num_filters, num_conv_layers)
        kernels = expandConfigList(filter_sizes, num_conv_layers)
        name = modelName(num_conv_layers, filters, kernels, pooling_type, False)
        save_path = os.path.join(weights_dir, f'{name}.keras')

        # Biar jelas aja
        print("\n" + "="*50)
        print(f"EXPERIMENT #{i+1}: {name}")
        print("-" * 50)
        print(f"Layers  : {num_conv_layers}")
        print(f"Filters : {filters}")
        print(f"Kernels : {kernels}")
        print(f"Pooling : {pooling_type}")
        print(f"Batch   : {batch_size} | Epochs: {epochs}")
        print("="*50 + "\n")

        model = CNN(
            input_shape=input_shape, 
            num_classes=num_classes, 
            num_conv_layers=num_conv_layers,
            num_filters=filters, 
            filter_sizes=kernels, 
            pooling_type=pooling_type
        )
        
        history = trainModel(
            model, X_train, y_train, X_val, y_val, 
            epochs=epochs, batch_size=batch_size, 
            save_path=save_path
        )
        
        metrics = evaluateKeras(model, X_test, y_test)
        
        results.append({
            'name': name, 
            'weights_path': save_path,
            'config': {
                'num_conv_layers': num_conv_layers, 
                'num_filters': filters,
                'filter_sizes': kernels, 
                'pooling_type': pooling_type
            },
            'history': history, 
            'metrics': metrics, 
            'params': int(model.count_params()),
        })
        
        with open(os.path.join(weights_dir, 'cnn_experiment_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    return results


def expandConfigList(values, length):
    if len(values) >= length:
        return list(values[:length])
    if not values:
        raise ValueError("Configuration list cannot be empty.")
    return list(values) + [values[-1]] * (length - len(values))


def modelName(num_conv_layers, num_filters, filter_sizes, pooling_type, use_locally_connected):
    layer_type = 'local' if use_locally_connected else 'conv'
    filters = '-'.join(str(v) for v in num_filters)
    kernels = '-'.join(str(v) for v in filter_sizes)
    return f'cnn_{layer_type}_L{num_conv_layers}_F{filters}_K{kernels}_P{pooling_type}'


class ScratchCNN:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X):
        return np.stack([self.forward(sample) for sample in X], axis=0)

    def countParams(self):
        total = 0
        for layer in self.layers:
            for attr in ('kernel', 'bias', 'W', 'b'):
                val = getattr(layer, attr, None)
                if val is not None:
                    total += int(np.prod(val.shape))
        return total


def scratchFromKeras(keras_model):
    scratch_layers = []
    for keras_layer in keras_model.layers:
        name = keras_layer.__class__.__name__
        if name == 'Conv2D':
            layer = Conv2D()
            layer.loadWeights(keras_layer)
        elif name == 'LocallyConnected2D':
            layer = LocallyConnected2D()
            layer.loadWeights(keras_layer)
        elif name == 'MaxPooling2D':
            layer = MaxPooling2D(pool_size=tuple(keras_layer.pool_size), strides=tuple(keras_layer.strides))
        elif name == 'AveragePooling2D':
            layer = AveragePooling2D(pool_size=tuple(keras_layer.pool_size), strides=tuple(keras_layer.strides))
        elif name == 'GlobalMaxPooling2D':
            layer = GlobalMaxPooling2D()
        elif name == 'GlobalAveragePooling2D':
            layer = GlobalAveragePooling2D()
        elif name == 'Flatten':
            layer = Flatten()
        elif name == 'Dense':
            layer = Dense()
            layer.loadWeights(keras_layer)
        elif name in ('InputLayer', 'Dropout', 'BatchNormalization'):
            continue
        else:
            raise ValueError(f"Unsupported Keras layer: {name}")
        scratch_layers.append(layer)
    return ScratchCNN(scratch_layers)
