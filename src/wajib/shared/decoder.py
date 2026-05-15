import numpy as np
from .preprocessing import END


# CNN feature + scratch model -> list of words (greedy)
def greedyDecode(rnn_scratch, proj_dense, embed_layer, out_dense, cnn_feature, vocab, max_len=30):
    id2word = {v: k for k, v in vocab.items()}
    END_TOKEN = vocab.get('<end>', 2) 

    h_states = [np.zeros(cell.hiddenDim) for cell in rnn_scratch.cells]
    c_states = [np.zeros(cell.hiddenDim) for cell in rnn_scratch.cells]
    x_img = proj_dense.forward(cnn_feature)
    h_states, c_states = rnn_scratch.forwardStep(x_img, h_states, c_states)

    words = []
    token = vocab['<start>']

    for _ in range(max_len):
        x_t = embed_layer.forward(token)
        h_states, c_states = rnn_scratch.forwardStep(x_t, h_states, c_states)
        h_final = h_states[-1] 
        logits = out_dense.forward(h_final)
        token = int(np.argmax(logits))
        
        if token == END_TOKEN or token == 0:
            break
            
        words.append(id2word.get(token, '<unk>'))

    return words