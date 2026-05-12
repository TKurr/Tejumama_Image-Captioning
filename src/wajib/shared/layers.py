
import numpy as np
from cnn.layers import applyActivation

# token-id => dense vector
class EmbeddingLayer:

    def __init__(self):
        self.embedding_matrix = None

    # load embedding matrix dari keras Embedding layer
    def loadWeights(self, keras_layer):
        self.embedding_matrix = np.asarray(keras_layer.get_weights()[0])

    # token ids (int atau array) jadi embedding vectors
    def forward(self, token_ids):
        return self.embedding_matrix[token_ids]

# linear projection + activation
class DenseLayer:

    def __init__(self, activation='linear'):
        self.W          = None
        self.b          = None
        self.activation = activation

    # load W, b dari keras Dense layer
    def loadWeights(self, keras_layer):
        weights         = keras_layer.get_weights()
        self.W          = np.asarray(weights[0])
        self.b          = np.asarray(weights[1]) if len(weights) > 1 else np.zeros(self.W.shape[-1])
        self.activation = getattr(keras_layer.activation, '__name__', self.activation)

    # x @ W + b lalu activation
    def forward(self, x):
        return applyActivation(x @ self.W + self.b, self.activation)