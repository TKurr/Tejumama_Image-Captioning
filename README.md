# Tugas Besar 2 Machine Learning - TejuMama

Proyek ini membuat model image captioning berbasis CNN + RNN/LSTM untuk menghasilkan deskripsi teks dari gambar. Repositori berisi pipeline pelatihan model, preprocessing data, serta notebook eksperimen.

## Fitur
- Ekstraksi fitur gambar dengan CNN (InceptionV3, from scratch).
- Pelatihan model RNN dan LSTM untuk sequence caption.
- Pipeline captioning end-to-end.
- Notebook analisis kualitatif hasil caption.

## Struktur Proyek
```
data/                Dataset dan utilitas unduh/ekstraksi
doc/                 Dokumen laporan
src/                 Kode sumber utama
src/notebook/        Notebook eksperimen
src/wajib/           Implementasi model, layer, dan utilitas
```

## Prasyarat
- Python 3.8+ (disarankan 3.10)
- pip

## Instalasi

**1. Buat dan aktifkan virtual environment, lalu install dependensi:**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Daftarkan venv ke Jupyter** (supaya kernel muncul di notebook):

```bash
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

## Dataset

Proyek menggunakan dua dataset: **Flickr8k** (image captioning) dan **Intel Image Classification** (CNN classification).

**3. Download kedua dataset** sekaligus via `init.py`:

Butuh akun Kaggle — `kagglehub` akan minta login otomatis jika belum ada credentials.

```bash
cd data && python init.py && cd ..
```

`init.py` mendownload via `kagglehub` dan membuat symlink di `data/`:

```
data/
├── seg_train/   # Intel Image Classification — training set (~14k gambar, 6 kelas)
├── seg_test/    # Intel Image Classification — test set (~3k gambar)
├── seg_pred/    # Intel Image Classification — unlabeled prediction set
└── flickr8k/    # Flickr8k — Images/ + captions.txt (8092 gambar)
```

**5. Ekstrak fitur Flickr8k** pakai InceptionV3 (cukup sekali, jalankan dari root proyek):

```bash
python data/extract_flickr_features.py
```

## Cara Menjalankan
Notebook tersedia di `src/notebook/` sesuai urutan proses:

1. `1_CNN_TrainingKeras.ipynb`
2. `2_CNN_ScratchNShared.ipynb`
3. `3_RNN_Training.ipynb`
4. `4_LSTM_Training.ipynb`
5. `5_Pipeline_Captioning.ipynb`
6. `6_QualitativeAnalysis.ipynb`

Jalankan notebook secara berurutan untuk mereplikasi hasil.

## Output
Beberapa output disimpan di:

- `src/wajib/weights/` *(Catatan: Sebagian besar output model berukuran besar tidak diupload ke GitHub. Silakan cek [Google Drive Berikut](https://drive.google.com/drive/folders/1iLIM7VGyiNLfDwnhBCbBG3Q0j3QmBQaJ) untuk mengunduh bobot selengkapnya. Beberapa model/gambar tetap disertakan di repositori ini).*

## Catatan
- Pastikan dataset sudah terunduh sebelum menjalankan notebook pelatihan.
- Waktu pelatihan tergantung GPU/CPU yang digunakan. *disarankan pakai GPU*
