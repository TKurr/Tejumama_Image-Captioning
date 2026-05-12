import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from shared.preprocessing import tokenizeCaption, padSequences



class RNNCell:
    # 1 step RNN
    # h_t = tanh(x_t @ Wx + h_prev @ Wh + b)

    def __init__(self):
        self.Wx         = None
        self.Wh         = None
        self.b          = None
        self.hidden_dim = None

    # load weights dari satu keras SimpleRNN layer
    def loadWeights(self, keras_layer):
        w           = keras_layer.get_weights()
        self.Wx         = w[0]
        self.Wh         = w[1]
        self.b          = w[2]
        self.hidden_dim = self.Wh.shape[0]

    def forward(self, x_t, h_prev):
        return np.tanh(x_t @ self.Wx + h_prev @ self.Wh + self.b)


# Stack of RNNCell for multi-layer RNN
class RNNScratch:

    def __init__(self):
        self.cells = []

    # load dari list keras SimpleRNN layers, satu cell per layer
    def loadWeights(self, keras_layers):
        for layer in keras_layers:
            cell = RNNCell()
            cell.loadWeights(layer)
            self.cells.append(cell)

    # (seq_len, input_dim) => (hidden_dim,) or (seq_len, hidden_dim)
    def forwardSequence(self, x_sequence, return_sequences=False):
        h       = [np.zeros(cell.hidden_dim) for cell in self.cells]
        outputs = []

        for t in range(len(x_sequence)):
            x = x_sequence[t]
            
            for i, cell in enumerate(self.cells):
                h[i] = cell.forward(x, h[i])
                x    = h[i]
                
            if return_sequences:
                outputs.append(h[-1].copy())

        return np.array(outputs) if return_sequences else h[-1]

    # satu timestep, carry h_states,  greedy decode step by step
    def forwardStep(self, x_t, h_states):
        x     = x_t
        new_h = []
        for i, cell in enumerate(self.cells):
            h_i = cell.forward(x, h_states[i])
            new_h.append(h_i)
            x   = h_i
        return new_h


# Build RNN Keras 
# Input: CNN feature + token ids jadi distribusi vocab di tiap timestep
def buildRNNKeras(vocab_size, embed_dim, hidden_dim, num_rnn_layers, cnn_feature_dim):
    cnn_input   = keras.Input(shape=(cnn_feature_dim,), name='cnn_feature')
    token_input = keras.Input(shape=(None,), dtype='int32', name='token_ids')

    projected = layers.Dense(embed_dim, name='cnn_proj')(cnn_input)
    projected = tf.expand_dims(projected, axis=1)

    embedded  = layers.Embedding(vocab_size, embed_dim, name='embedding')(token_input)

    x = tf.concat([projected, embedded], axis=1)

    for i in range(num_rnn_layers):
        x = layers.SimpleRNN(hidden_dim, return_sequences=True, name=f'rnn_{i}')(x)

    output = layers.Dense(vocab_size, activation='softmax', name='output')(x)

    model = keras.Model(inputs=[cnn_input, token_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model


# ====================== Data Prep and Training ======================

# Input: image features + captions 
# Output: numpy arrays (X_cnn, X_tokens, y)
def trainRNNDataset(image_features, captions_dict, vocab, max_len):
    X_cnn, X_tokens, Y = [], [], []

    for img_name, caps in captions_dict.items():
        if img_name not in image_features:
            continue
        feat = image_features[img_name]

        for cap in caps:
            token_ids    = tokenizeCaption(cap, vocab, max_len)
            input_tokens = [vocab['<start>']] + token_ids[:-1]

            X_cnn.append(feat)
            X_tokens.append(padSequences([input_tokens], max_len)[0])
            Y.append(padSequences([token_ids], max_len + 1)[0])

    return np.array(X_cnn), np.array(X_tokens), np.array(Y)


# Actual Training
# Input: arrays from trainRNNDataset + Keras model
# Output: train hist (loss per epoch)
def trainRNNKeras(
        model,
        X_cnn_train, X_tokens_train, y_train,
        X_cnn_val,   X_tokens_val,   y_val,
        epochs=20,
        batch_size=64,
        save_path=None,
):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    if save_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
        )

    history = model.fit(
        [X_cnn_train, X_tokens_train], y_train,
        validation_data=([X_cnn_val, X_tokens_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history.history
