import numpy as np
from typing import Tuple, Optional

# Fungsi Aktivasi
def __relu__(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def __softmax__(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def __linear__(x: np.ndarray) -> np.ndarray:
    return x


def applyActivation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation is None:
        return __linear__(x)

    activation = activation.lower()
    if activation == "relu":
        return __relu__(x)
    if activation == "softmax":
        return __softmax__(x)
    if activation in ("linear", "identity"):
        return __linear__(x)
    raise ValueError(f"Unsupported activation: {activation}")


# Konvolusi 2D (shared param)
class Conv2D:
    def __init__(self, activation: str = 'relu'):
        self.kernel = None
        self.bias = None
        self.strides = (1, 1)
        self.padding = 'valid'
        self.activation = activation

    def __loadWeights__(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        if len(weights) == 0:
            raise ValueError("Keras Conv2D layer has no weights to load.")

        self.kernel = np.asarray(weights[0])
        self.bias = np.asarray(weights[1]) if len(weights) > 1 else np.zeros(self.kernel.shape[-1])
        self.strides = tuple(keras_layer.strides)
        self.padding = keras_layer.padding.lower()
        self.activation = getattr(keras_layer.activation, "__name__", self.activation)

    def __padInput__(self, x: np.ndarray) -> np.ndarray:
        if self.padding == "valid":
            return x
        if self.padding != "same":
            raise ValueError(f"Unsupported padding: {self.padding}")

        h, w, _ = x.shape
        k_h, k_w, _, _ = self.kernel.shape
        s_h, s_w = self.strides

        out_h = int(np.ceil(h / s_h))
        out_w = int(np.ceil(w / s_w))
        pad_h = max((out_h - 1) * s_h + k_h - h, 0)
        pad_w = max((out_w - 1) * s_w + k_w - w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return np.pad(
            x,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            raise ValueError("Conv2D weights are not loaded.")

        x_pad = self.__padInput__(x)
        h_pad, w_pad, c_in = x_pad.shape
        k_h, k_w, k_c, c_out = self.kernel.shape
        s_h, s_w = self.strides

        if c_in != k_c:
            raise ValueError(f"Input channel mismatch: got {c_in}, expected {k_c}")

        out_h = (h_pad - k_h) // s_h + 1
        out_w = (w_pad - k_w) // s_w + 1
        output = np.zeros((out_h, out_w, c_out), dtype=np.float32)

        for i in range(out_h):
            row = i * s_h
            for j in range(out_w):
                col = j * s_w
                patch = x_pad[row:row + k_h, col:col + k_w, :]
                output[i, j, :] = np.tensordot(
                    patch,
                    self.kernel,
                    axes=((0, 1, 2), (0, 1, 2)),
                ) + self.bias

        return applyActivation(output, self.activation)


# Konvolusi 2D (tanpa shared)
class LocallyConnected2D:
    def __init__(self, activation: str = 'relu'):
        self.kernel = None
        self.bias = None
        self.activation = activation
        self.kernel_size = None
        self.strides = (1, 1)

    def __loadWeights__(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        if len(weights) == 0:
            raise ValueError("Keras LocallyConnected2D layer has no weights to load.")

        self.kernel = np.asarray(weights[0])
        self.bias = np.asarray(weights[1]) if len(weights) > 1 else np.zeros(
            (self.kernel.shape[0], self.kernel.shape[-1])
        )
        self.kernel_size = tuple(keras_layer.kernel_size)
        self.strides = tuple(keras_layer.strides)
        self.activation = getattr(keras_layer.activation, "__name__", self.activation)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            raise ValueError("LocallyConnected2D weights are not loaded.")

        h, w, c_in = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.strides
        out_h = (h - k_h) // s_h + 1
        out_w = (w - k_w) // s_w + 1
        c_out = self.kernel.shape[-1]
        output = np.zeros((out_h, out_w, c_out), dtype=np.float32)

        for idx in range(out_h * out_w):
            i, j = divmod(idx, out_w)
            row = i * s_h
            col = j * s_w
            patch = x[row:row + k_h, col:col + k_w, :].reshape(-1)
            output[i, j, :] = patch @ self.kernel[idx] + self.bias[idx]

        return applyActivation(output, self.activation)


class MaxPooling2D:
    def __init__(self, pool_size=(2,2), strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        h, w, c = x.shape
        p_h, p_w = self.pool_size
        s_h, s_w = self.strides
        out_h = (h - p_h) // s_h + 1
        out_w = (w - p_w) // s_w + 1
        output = np.zeros((out_h, out_w, c), dtype=x.dtype)

        for i in range(out_h):
            row = i * s_h
            for j in range(out_w):
                col = j * s_w
                patch = x[row:row + p_h, col:col + p_w, :]
                output[i, j, :] = np.max(patch, axis=(0, 1))

        return output

class AveragePooling2D:
    def __init__(self, pool_size=(2,2), strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        h, w, c = x.shape
        p_h, p_w = self.pool_size
        s_h, s_w = self.strides
        out_h = (h - p_h) // s_h + 1
        out_w = (w - p_w) // s_w + 1
        output = np.zeros((out_h, out_w, c), dtype=x.dtype)

        for i in range(out_h):
            row = i * s_h
            for j in range(out_w):
                col = j * s_w
                patch = x[row:row + p_h, col:col + p_w, :]
                output[i, j, :] = np.mean(patch, axis=(0, 1))

        return output


class GlobalMaxPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.max(x, axis=(0, 1))


class GlobalAveragePooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=(0, 1))

class Flatten:

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.flatten(order="C")

class Dense:
    def __init__(self, activation: str = 'linear'):
        self.W = None
        self.b = None
        self.activation = activation

    def __loadWeights__(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        if len(weights) == 0:
            raise ValueError("Keras Dense layer has no weights to load.")

        self.W = np.asarray(weights[0])
        self.b = np.asarray(weights[1]) if len(weights) > 1 else np.zeros(self.W.shape[-1])
        self.activation = getattr(keras_layer.activation, "__name__", self.activation)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise ValueError("Dense weights are not loaded.")
        return applyActivation(x @ self.W + self.b, self.activation)
