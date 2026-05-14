import numpy as np
from .preprocessing import END


# CNN feature + scratch model -> list of words (greedy)
def greedyDecode(rnn_scratch, proj_dense, embed_layer, out_dense, cnn_feature, vocab, max_len=30):
    id2word = {v: k for k, v in vocab.items()}

    x = proj_dense.forward(cnn_feature)
    h = [np.zeros(cell.hidden_dim) for cell in rnn_scratch.cells]
    h = rnn_scratch.forwardStep(x, h)

    words = []
    token = vocab['<start>']

    for _ in range(max_len):
        x     = embed_layer.forward(token)
        h     = rnn_scratch.forwardStep(x, h)
        token = int(np.argmax(out_dense.forward(h[-1])))
        if token == END:
            break
        words.append(id2word.get(token, '<unk>'))

    return words
