import numpy as np 
from typing import Dict, List, Tuple, Optional 

class LSTMCell: 
    
    def __init__(self):
        self.W_x = None
        self.W_h = None
        self.b = None
        self.hidden_dim = None 

    def __loadWeights__(self, keras_lstm_layer, layer_index: int = 0) -> None:
        pass

    def forward(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

class LSTMScratch: 
    def __init__(self):
        self.cells: List[LSTMCell] = []

    def __loadWeights__(self, keras_layers: list) -> None:
        pass

    def forwardSequence(self,
        x_sequence: np.ndarray, 
        return_sequences: bool = False 
    ) -> np.ndarray:
        pass

# helper func
def __buildLSTM_Keras__( 
    vocab_size: int, 
    embed_dim: int, 
    hidden_dim: int, 
    num_lstm_layers: int,
    cnn_feature_dim: int, 
) -> 'keras.Model':
    pass 

def __trainLSTM_Keras__( 
    model, 
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    epochs: int = 20,
    batch_size: int = 64,
    save_path: Optional[str] = None,
) -> dict:
    pass 

class ImageCaptioningPipleline: 
    def __init__(
        self, 
        cnn_encoder,
        project_dense,
        embedding_layer,
        lstm_scratch, 
        output_dense, 
        vocab: Dict[str, int], 
        id2word: Dict[int, str],
        target_size: Tuple[int, int] = (299, 299), 
        max_len: int = 30, 
    ): 
        pass

    def generateCaption(self, image_path: str) -> str: 
        pass

    def __evaluate_LSTM__(
            self,
            test_image_paths: List[str],
            reference_dict: Dict[str, List[List[str]]],
    ) -> Dict[str, float]:
        pass
