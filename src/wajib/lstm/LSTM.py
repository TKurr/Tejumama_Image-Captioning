import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image

from ..shared.preprocessing import tokenizeCaption, padSequences


class LSTMCell:
    def __init__(self):
        self.wx = None
        self.wh = None
        self.bias = None
        self.hiddenDim = None

    def loadWeights(self, kerasLayer) -> None:
        weights = kerasLayer.get_weights()
        self.wx = weights[0]
        self.wh = weights[1]
        bias = weights[2]
        
        if len(bias.shape) == 2:
            self.bias = np.sum(bias, axis=0)
        else:
            self.bias = bias

        self.hiddenDim = self.wh.shape[0]

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(
        self,
        xT: np.ndarray,
        hPrev: np.ndarray,
        cPrev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        gates = xT @ self.wx + hPrev @ self.wh + self.bias

        hiddenDim = self.hiddenDim
        inputGate = self.sigmoid(gates[:hiddenDim])
        forgetGate = self.sigmoid(gates[hiddenDim:2 * hiddenDim])
        candidateGate = np.tanh(gates[2 * hiddenDim:3 * hiddenDim])
        outputGate = self.sigmoid(gates[3 * hiddenDim:4 * hiddenDim])

        cT = forgetGate * cPrev + inputGate * candidateGate
        hT = outputGate * np.tanh(cT)

        return hT, cT


class LSTMScratch:
    def __init__(self):
        self.cells: List[LSTMCell] = []

    def loadWeights(self, kerasLayers: list) -> None:
        for layer in kerasLayers:
            cell = LSTMCell()
            cell.loadWeights(layer)
            self.cells.append(cell)

    def forwardSequence(
        self,
        xSequence: np.ndarray,
        returnSequences: bool = False,
    ) -> np.ndarray:
        hStates = [np.zeros(cell.hiddenDim) for cell in self.cells]
        cStates = [np.zeros(cell.hiddenDim) for cell in self.cells]
        outputs = []
        for t in range(len(xSequence)):
            x = xSequence[t]
            for i, cell in enumerate(self.cells):
                hStates[i], cStates[i] = cell.forward(x, hStates[i], cStates[i])
                x = hStates[i]

            if returnSequences:
                outputs.append(hStates[-1].copy())

        return np.array(outputs) if returnSequences else hStates[-1]

    def forwardStep(
        self,
        xT: np.ndarray,
        hStates: list,
        cStates: list,
    ) -> Tuple[list, list]:
        x = xT
        newH, newC = [], []

        for i, cell in enumerate(self.cells):
            hI, cI = cell.forward(x, hStates[i], cStates[i])
            newH.append(hI)
            newC.append(cI)
            x = hI

        return newH, newC


def buildLSTMKeras(
    vocabSize: int,
    embedDim: int,
    hiddenDim: int,
    numLstmLayers: int,
    cnnFeatureDim: int,
) -> keras.Model:
    cnnInput = keras.Input(shape=(cnnFeatureDim,), name='cnn_feature')
    tokenInput = keras.Input(shape=(None,), dtype='int32', name='token_ids')
    projected = layers.Dense(embedDim, activation='relu', name='cnn_proj')(cnnInput)
    projected = keras.ops.expand_dims(projected, axis=1)
    embedded = layers.Embedding(vocabSize, embedDim, name='embedding')(tokenInput)
    x = keras.ops.concatenate([projected, embedded], axis=1)
    for i in range(numLstmLayers):
        x = layers.LSTM(hiddenDim, return_sequences=True, implementation=1, name=f'lstm_{i}')(x)
        if i < numLstmLayers - 1:
            x = layers.Dropout(0.3, name=f'dropout_{i}')(x)
    output = layers.Dense(vocabSize, name='output')(x)
    model = keras.Model(inputs=[cnnInput, tokenInput], outputs=output)
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0),
    )
    
    return model


def trainLSTMDataset(imageFeatures, captionsDict, vocab, maxLen):
    xCnn = []
    xTokens = []
    y = []

    for imgName, caps in captionsDict.items():
        if imgName not in imageFeatures:
            continue
        feat = imageFeatures[imgName]
        for cap in caps:
            tokenIds = tokenizeCaption(cap, vocab, maxLen)
            inputTokens = [vocab['<start>']] + tokenIds[:-1]
            xCnn.append(feat)
            xTokens.append(padSequences([inputTokens], maxLen)[0])
            y.append(padSequences([tokenIds], maxLen + 1)[0])

    return np.array(xCnn), np.array(xTokens), np.array(y)

def trainLSTMKeras(
    model,
    xCnnTrain, xTokensTrain, yTrain,
    xCnnVal, xTokensVal, yVal,
    epochs: int = 20,
    batchSize: int = 64,
    savePath: Optional[str] = None,
) -> dict:
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
    ]
    if savePath:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                savePath, monitor='val_loss', save_best_only=True
            )
        )

    history = model.fit(
        [xCnnTrain, xTokensTrain], yTrain,
        validation_data=([xCnnVal, xTokensVal], yVal),
        epochs=epochs,
        batch_size=batchSize,
        callbacks=callbacks,
        verbose=1,
    )
    
    return history.history


class ImageCaptioningPipeline:
    def __init__(
        self,
        cnnEncoder,
        projectDense,
        embeddingLayer,
        lstmScratch,
        outputDense,
        vocab: Dict[str, int],
        id2word: Dict[int, str],
        targetSize: Tuple[int, int] = (299, 299),
        maxLen: int = 30,
    ):
        self.cnnEncoder = cnnEncoder
        self.projDense = projectDense
        self.embedLayer = embeddingLayer
        self.lstmScratch = lstmScratch
        self.outDense = outputDense
        self.vocab = vocab
        self.id2word = id2word
        self.targetSize = targetSize
        self.maxLen = maxLen

    def loadImage(self, imagePath: str) -> np.ndarray:
        img = Image.open(imagePath).convert('RGB')
        img = img.resize(self.targetSize)
        img = np.array(img) / 255.0
        return img

    def generateCaption(self, imagePath: str) -> str:
        img = self.loadImage(imagePath)
        imgBatch = np.expand_dims(img, axis=0)
        cnnFeature = self.cnnEncoder.predict(imgBatch, verbose=0)[0]
        x = self.projDense.forward(cnnFeature)
        h = [np.zeros(cell.hiddenDim) for cell in self.lstmScratch.cells]
        c = [np.zeros(cell.hiddenDim) for cell in self.lstmScratch.cells]
        words = []
        token = self.vocab['<start>']
        for _ in range(self.maxLen):
            xEmb = self.embedLayer.forward(token)
            h, c = self.lstmScratch.forwardStep(xEmb, h, c)
            logits = self.outDense.forward(h[-1])
            token = int(np.argmax(logits))
            if token == self.vocab['<end>']:
                break
            words.append(self.id2word.get(token, '<unk>'))
        return ' '.join(words)

    def evaluateLSTM(
        self,
        testImagePaths: List[str],
        referencesDict: Dict[str, List[List[str]]],
    ) -> Dict[str, float]:
        from nltk.translate.bleu_score import corpus_bleu
        hypotheses = []
        references = []

        for imgPath in testImagePaths:
            imgName = Path(imgPath).stem
            if imgName not in referencesDict:
                continue
            caption = self.generateCaption(imgPath)
            hypotheses.append(caption.split())
            refs = referencesDict[imgName]
            references.append(refs)

        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        return {'bleu4': bleu4}
