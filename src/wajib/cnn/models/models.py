import numpy as np
import json
import os
from itertools import product
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score

from ..layers.layers import (
    AveragePooling2D, Conv2D, Dense, Flatten,
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    LocallyConnected2D, MaxPooling2D,
)


@keras.utils.register_keras_serializable(package='wajib')
class KerasLocallyConnected2D(layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            padding='valid',
            activation=None,
            use_bias=True,
            kernel_regularizer=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.kernel_size = tuple(int(v) for v in self.kernel_size)
        self.strides = tuple(strides) if isinstance(strides, (list, tuple)) else (strides, strides)
        self.strides = tuple(int(v) for v in self.strides)
        self.padding = padding.lower()
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.out_h = None
        self.out_w = None

    def build(self, input_shape):
        _, input_h, input_w, input_c = input_shape
        if input_h is None or input_w is None or input_c is None:
            raise ValueError("KerasLocallyConnected2D needs a fully defined input height, width, and channels.")
        self.out_h, self.out_w, _ = self.compute_output_shape(input_shape)[1:]
        patch_size = self.kernel_size[0] * self.kernel_size[1] * int(input_c)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.out_h * self.out_w, patch_size, self.filters),
            initializer='glorot_uniform',
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.out_h * self.out_w, self.filters),
                initializer='zeros',
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        _, input_h, input_w, _ = input_shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.strides
        if self.padding == 'same':
            out_h = int(np.ceil(input_h / s_h))
            out_w = int(np.ceil(input_w / s_w))
        elif self.padding == 'valid':
            out_h = (input_h - k_h) // s_h + 1
            out_w = (input_w - k_w) // s_w + 1
        else:
            raise ValueError(f"Unsupported padding: {self.padding}")
        return (input_shape[0], out_h, out_w, self.filters)

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding.upper(),
        )
        patches = tf.reshape(patches, [tf.shape(inputs)[0], self.out_h * self.out_w, -1])
        output = tf.einsum('bip,ipf->bif', patches, self.kernel)
        if self.bias is not None:
            output = output + self.bias
        output = tf.reshape(output, [tf.shape(inputs)[0], self.out_h, self.out_w, self.filters])
        return self.activation(output) if self.activation is not None else output

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


def CNN(
        input_shape, 
        num_classes, 
        num_conv_layers, 
        num_filters,
        filter_sizes,
        pooling_type,
        use_locally_connected=False,
        locally_connected_layers='all',
        use_global_pooling=False,
        l2_strength=0.0,
        dense_units=256,
    ):
    if num_conv_layers < 1:
        raise ValueError("num_conv_layers must be at least 1.")
    if pooling_type not in ('max', 'average'):
        raise ValueError("pooling_type must be 'max' or 'average'.")

    filters = expandConfigList(num_filters, num_conv_layers)
    kernels = expandConfigList(filter_sizes, num_conv_layers)
    kernel_regularizer = keras.regularizers.l2(l2_strength) if l2_strength else None

    if use_locally_connected is True:
        local_mode = locally_connected_layers
    elif use_locally_connected in ('all', 'last', 'none'):
        local_mode = use_locally_connected
        use_locally_connected = local_mode != 'none'
    else:
        local_mode = 'none'

    if local_mode not in ('all', 'last', 'none'):
        raise ValueError("locally_connected_layers must be 'all', 'last', or 'none'.")

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for idx in range(num_conv_layers):
        is_local = use_locally_connected and (local_mode == 'all' or idx == num_conv_layers - 1)
        conv_layer = KerasLocallyConnected2D if is_local else layers.Conv2D
        model.add(conv_layer(
            filters[idx],
            kernels[idx],
            padding='same',
            activation='relu',
            kernel_regularizer=kernel_regularizer,
        ))
        model.add(layers.BatchNormalization()) 

        if pooling_type == 'max':
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        
        model.add(layers.Dropout(0.2)) 

    if use_global_pooling:
        model.add(layers.GlobalAveragePooling2D())
    else:
        model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu', kernel_regularizer=kernel_regularizer)) 
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
        [2, 3],                      
        [[32, 64], [64, 128]],       
        [[3, 3], [5, 5]],           
        ['max', 'average']           
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
        elif name in ('LocallyConnected2D', 'KerasLocallyConnected2D'):
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
