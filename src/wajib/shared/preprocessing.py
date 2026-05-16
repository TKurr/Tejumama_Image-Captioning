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

def cleanCaption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", '', caption)
    return re.sub(r'\s+', ' ', caption).strip()

def loadFlickr8kCaptions(captions_file):
    df = pd.read_csv(captions_file)
    result = {}
    for _, row in df.iterrows():
        result.setdefault(row['image'], []).append(cleanCaption(row['caption']))
    return result

def buildVocabulary(captions, min_freq=1):
    counter = Counter(w for cap in captions for w in cap.split())
    vocab = dict(SPECIAL)
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab

def saveVocabulary(vocab, path):
    with open(path, 'w') as f:
        json.dump(vocab, f)

def loadVocabulary(path):
    with open(path) as f:
        return {k: int(v) for k, v in json.load(f).items()}

def tokenizeCaption(caption, vocab, max_len):
    tokens = caption.split()[: max_len - 1]
    return [vocab.get(w, UNK) for w in tokens] + [END]

def padSequences(sequences, max_len, pad_id=0):
    out = np.full((len(sequences), max_len), pad_id, dtype=np.int32)
    for i, seq in enumerate(sequences):
        n = min(len(seq), max_len)
        out[i, :n] = seq[:n]
        
    return out