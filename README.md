# Tugas Besar 2 Machine Learning - TejuMama

Proyek ini membuat model image captioning berbasis CNN + RNN/LSTM untuk menghasilkan deskripsi teks dari gambar. Repositori berisi pipeline pelatihan model, preprocessing data, serta notebook eksperimen.

## Fitur
- Ekstraksi fitur gambar dengan CNN.
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
1. Buat dan aktifkan virtual environment.
2. Install dependensi:

```
pip install -r requirements.txt
```

## Dataset
Proyek menggunakan dataset Flickr8k. Script unduh tersedia di:

- `data/download_flickr.py`
- `data/extract_flickr_features.py`

Contoh alur singkat:

```
python data/download_flickr.py
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
2 
## Catatan
- Pastikan dataset sudah terunduh sebelum menjalankan notebook pelatihan.
- Waktu pelatihan tergantung GPU/CPU yang digunakan. *disarankan pakai GPU*

