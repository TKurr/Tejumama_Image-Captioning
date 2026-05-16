# CNN & RNN/LSTM Image Captioning

Implementasi CNN, RNN, dan LSTM from scratch pakai NumPy, dengan Keras untuk training.

> Image captioning menggabungkan computer vision dan NLP: CNN encoder mengekstrak fitur visual dari gambar, lalu recurrent decoder (RNN atau LSTM) menghasilkan caption kata per kata.

![Dataset](https://img.shields.io/badge/Dataset-Intel%20Image%20Classification%20%2B%20Flickr8k-blue)
![Task](https://img.shields.io/badge/Task-Image%20Classification%20%2B%20Image%20Captioning-orange)

## **Overview**

Forward propagation CNN, SimpleRNN, dan LSTM diimplementasi from scratch (NumPy only), training pakai Keras. Komponen utama:

- **CNN**: Conv2D (shared), LocallyConnected2D (non-shared), Pooling, Flatten, Dense. Eksperimen 16 arsitektur pada dataset Intel Image Classification.
- **RNN/LSTM decoder**: arsitektur pre-inject (Show and Tell, Vinyals et al. 2015), 12 variasi training (6 RNN + 6 LSTM).
- **Image captioning pipeline**: dari raw image sampai caption string, lewat InceptionV3 features dan RNN/LSTM decoder.
- **Evaluasi**: macro F1-score (CNN), BLEU-4 + METEOR (captioning), perbandingan Keras vs scratch.

## **Setup**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download dataset (butuh Kaggle credentials):

```bash
cd data && python init.py && cd ..
```

Extract fitur gambar Flickr8k pakai InceptionV3 (cukup sekali):

```bash
python data/extract_flickr_features.py
```

Setelah itu jalankan notebook sesuai urutan. Berikut di mana tiap file disimpan dan dipakai:

```py
# Feature extraction  (data/extract_flickr_features.py)
src/wajib/weights/features/flickr8k_features.npy   # fitur InceptionV3 semua gambar Flickr8k
src/wajib/weights/features/flickr8k_index.json     # mapping filename ke index array

# Vocabulary  (dibuat saat pertama kali run notebook 3 atau 4, dipakai bersama)
src/wajib/weights/vocab.json                        # mapping kata ke int id, diload notebook 3-6

# CNN training  (notebook 1)
src/wajib/weights/cnn/<name>.keras                  # 16 model weights
src/wajib/weights/cnn/cnn_experiment_results.json   # F1-score + config tiap arsitektur

# RNN training  (notebook 3)
src/wajib/weights/rnn/rnn_<L>L_<H>h.keras          # 6 model weights (contoh: rnn_1L_128h.keras)

# LSTM training  (notebook 4)
src/wajib/weights/lstm/lstm_<L>L_<H>h.keras        # 6 model weights (contoh: lstm_1L_128h.keras)
```

Notebook 5 load `vocab.json`, `.npy` features, dan `.keras` terbaik dari `rnn/` atau `lstm/` untuk jalankan captioning end-to-end dan hitung BLEU-4.

**Struktur proyek:**

```sh
Tejumama_Image-Captioning
├── data/                          # datasets (tidak di-commit)
├── doc/                           # laporan PDF
├── requirements.txt
└── src/
    ├── notebook/
    │   ├── 1_CNN_TrainingKeras.ipynb       # training 16 arsitektur CNN
    │   ├── 2_CNN_ScratchNShared.ipynb      # forward scratch + evaluasi CNN
    │   ├── 3_RNN_Training.ipynb            # training 6 variasi RNN
    │   ├── 4_LSTM_Training.ipynb           # training 6 variasi LSTM
    │   ├── 5_Pipeline_Captioning.ipynb     # full pipeline + eksperimen
    │   └── 6_QualitativeAnalysis.ipynb     # qualitative analysis (10 sampel)
    └── wajib/
        ├── cnn/
        │   ├── layers/layers.py            # Conv2D, LocallyConnected2D, Pooling, Flatten, Dense
        │   ├── models/models.py            # CNN builder, train, evaluate
        │   └── utils/utils.py             # image_loader, batch_loader, feature_extractor
        ├── rnn/
        │   └── RNN.py                     # RNNCell, RNNScratch, buildRNN_Keras
        ├── lstm/
        │   └── LSTM.py                    # LSTMCell, LSTMScratch, buildLSTM_Keras, pipeline
        ├── shared/
        │   ├── layers.py                  # EmbeddingLayer, DenseLayer (dipakai RNN & LSTM)
        │   ├── preprocessing.py           # caption cleaning, vocab, tokenization, padding
        │   └── decoder.py                 # greedyDecode, computeBLEU4, computeMETEOR
        └── weights/                       # bobot model hasil training (tidak di-commit)
```

---

### Author

| NIM      | Name                | Contribution                                                                                                                                                                                   |
| :------- | :------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 13523148 | Andrew Tedjapratama | Implementasi RNN, training dan inference RNN. Shared layers and preprocessing RNN, feature extraction encoding dan decoding, Pipeline Captioning. Pengerjaan Laporan, Eksperimen dan Evaluasi. |
| 13523154 | Theo Kurniady       | Implementasi CNN, training dan inference CNN, Qualitative Analysis Perbandingan. Implementasi Utility Functions. Pengerjaan Laporan, Eksperimen dan Evaluasi.                                  |
| 13523158 | Lukas Raja Agripa   | Implementasi LSTM, training dan inference LSTM, Qualitative Analysis Perbandingan. Setup initial code architecture. Pengerjaan Laporan, Eksperimen dan Evaluasi.                               |

## Referensi

- [d2l.ai: RNN chapter](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
- [d2l.ai: LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html)
- [d2l.ai: Deep RNN](https://d2l.ai/chapter_recurrent-modern/deep-rnn.html)
- [d2l.ai: Bidirectional RNN](https://d2l.ai/chapter_recurrent-modern/bi-rnn.html)
- [d2l.ai: CNN chapter](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
- [d2l.ai: Image Captioning](https://d2l.ai/chapter_computer-vision/image-captioning.html)
- [d2l.ai: Beam Search](https://d2l.ai/chapter_recurrent-modern/beam-search.html)
- [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
- [Vinyals et al. (2015): Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Tanti et al. (2017): Where to put the Image in an Image Caption Generator](https://arxiv.org/abs/1703.09137)
- [Selvaraju et al. (2016): Grad-CAM](https://arxiv.org/abs/1610.02391)
- [Flickr8k dataset (Kaggle)](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [Intel Image Classification dataset (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- [Karpathy: The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [CS231n Lecture 10: RNNs & Image Captioning (Stanford)](https://cs231n.stanford.edu/slides/2024/lecture_10.pdf)

## Acknowledgements

- Dosen IF3270 Pembelajaran Mesin, Institut Teknologi Bandung, 2026
- Asisten IF3270 Pembelajaran Mesin, Institut Teknologi Bandung, 2026
