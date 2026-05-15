import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..shared.preprocessing import tokenizeCaption, padSequences
class RNNCell:
    def __init__(self, mode='rnn'):
        self.Wx = None
        self.Wh = None
        self.b  = None
        self.mode = mode # 'rnn' atau 'lstm'

    def loadWeights(self, keras_layer):
        w = keras_layer.get_weights()
        self.Wx = w[0]
        self.Wh = w[1]
        self.b  = w[2]
        self.hidden_dim = self.Wh.shape[0]
        # Deteksi otomatis tipe layer dari Keras
        if 'lstm' in keras_layer.name.lower():
            self.mode = 'lstm'

    def forward(self, x_t, h_prev, c_prev=None):
        if self.mode == 'rnn':
            h_next = np.tanh(x_t @ self.Wx + h_prev @ self.Wh + self.b)
            return h_next, None # Tetap return tuple biar konsisten
        
        else: # Logika LSTM
            # Hitung semua gate sekaligus (dimensi akan jadi 4 * hidden_dim)
            z = x_t @ self.Wx + h_prev @ self.Wh + self.b
            
            # Potong jadi 4 bagian: input, forget, cell_gate, output
            i, f, g, o = np.split(z, 4, axis=-1)
            
            # Terapkan aktivasi
            i = 1 / (1 + np.exp(-i)) # sigmoid
            f = 1 / (1 + np.exp(-f)) # sigmoid
            g = np.tanh(g)
            o = 1 / (1 + np.exp(-o)) # sigmoid
            
            c_next = f * c_prev + i * g
            h_next = o * np.tanh(c_next)
            return h_next, c_next

class RNNScratch:
    # Stack of RNNCell for multi-layer RNN

    def __init__(self):
        self.cells = []

    # load dari list keras SimpleRNN layers, satu cell per layer
    def loadWeights(self, keras_layers):
        for layer in keras_layers:
            cell = RNNCell()
            cell.loadWeights(layer)
            self.cells.append(cell)

    def forwardStep(self, x_t, states):
        x     = x_t
        new_states = []
        
        for i, cell in enumerate(self.cells):
            h_prev, c_prev = states[i]
            h_i, c_i = cell.forward(x, h_prev, c_prev)
            new_states.append((h_i, c_i))
            x = h_i 
            
        return new_states

    def forwardSequence(self, x_sequence, return_sequences=False):
        states = [(np.zeros(cell.hidden_dim), np.zeros(cell.hidden_dim)) for cell in self.cells]
        outputs = []

        for t in range(len(x_sequence)):
            x = x_sequence[t]
            states = self.forwardStep(x, states)
            h_last_layer = states[-1][0]
            
            if return_sequences:
                outputs.append(h_last_layer.copy())

        return np.array(outputs) if return_sequences else states[-1][0]


def buildRNNKeras(vocab_size, embed_dim, hidden_dim, num_rnn_layers, cnn_feature_dim, rnn_type='rnn'):
    cnn_input   = keras.Input(shape=(cnn_feature_dim,), name='cnn_feature')
    token_input = keras.Input(shape=(None,), dtype='int32', name='token_ids')

    projected = layers.Dense(embed_dim, activation='relu', name='cnn_proj')(cnn_input)
    projected = layers.Reshape((1, embed_dim))(projected)

    embedded  = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name='embedding')(token_input)

    x = layers.Concatenate(axis=1)([projected, embedded])

    for i in range(num_rnn_layers):
        if rnn_type.lower() == 'lstm':
            x = layers.LSTM(hidden_dim, return_sequences=True, name=f'lstm_{i}')(x)
        else:
            x = layers.SimpleRNN(hidden_dim, return_sequences=True, name=f'rnn_{i}')(x)
        x = layers.Dropout(0.3)(x)

    output = layers.Dense(vocab_size, activation='softmax', name='output')(x)

    model = keras.Model(inputs=[cnn_input, token_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# ====================== Data Prep and Training ======================

# Input: image features + captions
# Output: numpy arrays (X_cnn, X_tokens, y)
def trainRNNDataset(image_features, captions_dict, vocab, max_len):
    X_cnn = [] 
    X_tokens = []
    Y = []

    for img_name, caps in captions_dict.items():
        if img_name not in image_features:
            continue
        feat = image_features[img_name]

        for cap in caps:
            token_ids = tokenizeCaption(cap, vocab, max_len)
            
            input_tokens = [vocab['<start>']] + token_ids[:-1]
            
            X_cnn.append(feat)
            X_tokens.append(padSequences([input_tokens], max_len)[0])
            Y.append(padSequences([[vocab['<start>']] + token_ids], max_len + 1)[0])

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
