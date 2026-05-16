import numpy as np
from .preprocessing import END

def greedyDecode(rnn_scratch, proj_dense, embed_layer, out_dense, cnn_feature, vocab, max_len=30):
    id2word = {v: k for k, v in vocab.items()}
    END_TOKEN = vocab.get('<end>', 2) 

    hidden_dim = getattr(rnn_scratch.cells[0], 'hiddenDim', getattr(rnn_scratch.cells[0], 'hidden_dim', 0))
    is_rnn = rnn_scratch.__class__.__name__ == 'RNNScratch'

    if is_rnn:
        states = [(np.zeros(hidden_dim), np.zeros(hidden_dim)) for _ in rnn_scratch.cells]
        x_img = proj_dense.forward(cnn_feature)
        states = rnn_scratch.forwardStep(x_img, states)

        words = []
        token = vocab['<start>']

        for _ in range(max_len):
            x_t = embed_layer.forward(token)
            states = rnn_scratch.forwardStep(x_t, states)
            h_final = states[-1][0]
            
            logits = out_dense.forward(h_final)
            token = int(np.argmax(logits))
            
            if token == END_TOKEN or token == 0:
                break
                
            words.append(id2word.get(token, '<unk>'))
        return words

    else:
        hStates = [np.zeros(hidden_dim) for _ in rnn_scratch.cells]
        cStates = [np.zeros(hidden_dim) for _ in rnn_scratch.cells]

        x_img = proj_dense.forward(cnn_feature)
        hStates, cStates = rnn_scratch.forwardStep(x_img, hStates, cStates)

        words = []
        token = vocab['<start>']

        for _ in range(max_len):
            x_t = embed_layer.forward(token)
            hStates, cStates = rnn_scratch.forwardStep(x_t, hStates, cStates)
            
            h_final = hStates[-1]
            
            logits = out_dense.forward(h_final)
            token = int(np.argmax(logits))
            
            if token == END_TOKEN or token == 0:
                break
                
            words.append(id2word.get(token, '<unk>'))

        return words