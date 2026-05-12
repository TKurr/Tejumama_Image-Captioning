import re
import json
from collections import Counter
import numpy as np
import pandas as pd

PAD, START, END, UNK = 0, 1, 2, 3
SPECIAL =   {
            '<pad>': PAD,
            '<start>': START,
            '<end>': END,
            '<unk>': UNK
        }


# caption string => lowercase, hapus punctuation
def __cleanCaption__(caption):
    caption = caption.lower()
    
    caption = re.sub(r"[^a-z0-9\s]", '', caption)
    
    return re.sub(r'\s+', ' ', caption).strip()


# baca captions.txt flickr8k, => {image_name: [cap1, cap2, ...]}
def __loadFlickr8kCaptions__(captions_file):
    df = pd.read_csv(captions_file)
    result = {}
    
    for _, row in df.iterrows():
        result.setdefault(row['image'], []).append(__cleanCaption__(row['caption']))
        
    return result


# list of caption strings 
# contoh: {word: id}, special tokens always di depan (0-3)
def __buildVocabulary__(captions, min_freq=1):
    counter = Counter(w for cap in captions for w in cap.split())
    vocab = dict(SPECIAL)
    
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
            
    return vocab

# vocab dict => file json
def __saveVocabulary__(vocab, path):
    with open(path, 'w') as f:
        json.dump(vocab, f)


# file json => vocab dict
def __loadVocabulary__(path):
    with open(path) as f:
        return {k: int(v) for k, v in json.load(f).items()}


# caption string jadi list token ids, 
# diakhiri <end>, tanpa <start>, max_len
def __tokenizeCaption__(caption, vocab, max_len):
    tokens = caption.split()[: max_len - 1]
    
    return [vocab.get(w, UNK) for w in tokens] + [END]


# list of token id lists jadi ndarray (N, max_len), 
# kl kelebihan dipotong, kekurangan dipad 0
def __padSequences__(sequences, max_len, pad_id=0):
    out = np.full((len(sequences), max_len), pad_id, dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        n = min(len(seq), max_len)
        out[i, :n] = seq[:n]
        
    return out