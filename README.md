# 🎭 Emotion Detection System

Sistem deteksi emosi wajah menggunakan computer vision dan machine learning untuk mendeteksi ekspresi **Marah**, **Sedih**, dan **Senang**.

## 📁 Struktur Project

```
emotion-detection/
├── 📄 app.py                          # Flask web application
├── 📄 train_model.py                  # Script training model
├── 📄 requirements.txt                # Dependencies Python
├── 📄 README.md                       # Dokumentasi utama
├── 📄 CARA_TINGKATKAN_AKURASI.md     # Panduan meningkatkan akurasi 80%+
├── 📄 .gitignore                      # Git ignore file
│
├── 📂 data2/                          # Dataset folder
│   ├── 📂 marah/                      # Gambar ekspresi marah (110 images)
│   ├── 📂 sedih/                      # Gambar ekspresi sedih (112 images)
│   ├── 📂 senang/                     # Gambar ekspresi senang (105 images)
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_eye.xml
│   └── haarcascade_mcs_mouth.xml
│
├── 📂 templates/                      # HTML templates
│   └── index.html                     # Web interface
│
├── 📂 uploads/                        # Folder untuk gambar yang diupload
│
├── 📂 .venv/                          # Virtual environment (auto-generated)
│
└── 📦 emotion_model.pkl              # Trained model (generated after training)
```

## � Fitur Utama

✅ **3 Kategori Emosi**: Marah, Sedih, Senang  
✅ **35 Fitur Ekstraksi**: Mata, mulut, tekstur, edge detection, histogram, gradien  
✅ **Machine Learning**: Random Forest / KNN dengan ensemble voting  
✅ **Web Interface**: Responsive design dengan drag & drop upload  
✅ **Visualisasi Keypoints**: Titik-titik mata dan mulut ditandai dengan warna  
✅ **Confidence Score**: Probabilitas untuk setiap kategori emosi  
✅ **Penjelasan Detail**: Alasan mengapa emosi terdeteksi berdasarkan koordinat fitur  
✅ **Real-time Detection**: Upload gambar dan dapatkan hasil instant

## 🎯 Model Performance

- **Akurasi**: 70.31% (dengan 317 gambar training)
- **Marah**: Precision 67%, Recall 55%
- **Sedih**: Precision 58%, Recall 64%
- **Senang**: Precision 86%, Recall 95% ⭐

> 💡 **Ingin akurasi 80%+?** Lihat [CARA_TINGKATKAN_AKURASI.md](CARA_TINGKATKAN_AKURASI.md)

## 🚀 Quick Start

### 1️⃣ Instalasi Dependencies

```bash
# Clone atau download project
cd emotion-detection

# Aktifkan virtual environment (jika belum)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Persiapkan Dataset

Dataset sudah tersedia di folder `data2/`:
```
data2/
├── marah/         # 108 images ✅
├── sedih/         # 109 images ✅
└── senang/        # 100 images ✅
```

> 💡 **Tips**: Untuk akurasi lebih tinggi, tambahkan lebih banyak gambar (target: 200-300/kategori)

### 3️⃣ Training Model

```bash
# Jalankan script training
python train_model.py
```

**Output**:
- Model tersimpan di: `emotion_model.pkl`
- Akurasi dan classification report ditampilkan
- Training time: ~30 detik

### 4️⃣ Jalankan Web Application

```bash
# Jalankan Flask server
python app.py
```

**Akses**: http://localhost:5000

### 5️⃣ Gunakan Aplikasi

1. **Upload gambar** wajah (drag & drop atau klik)
2. **Klik** "Analisis Emosi"
3. **Lihat hasil**:
   - ✅ Emosi yang terdeteksi (Marah/Sedih/Senang)
   - ✅ Confidence score (%)
   - ✅ Gambar dengan keypoints (titik mata & mulut berwarna)
   - ✅ Koordinat X, Y untuk setiap titik
   - ✅ Probability bars untuk semua emosi
   - ✅ Penjelasan detail kenapa emosi terdeteksi

## 🔧 Cara Kerja Sistem

### Ekstraksi 35 Fitur Wajah:

1. **Koordinat Wajah** (4 fitur): Posisi dan ukuran wajah
2. **Mata** (4 fitur): Jumlah, jarak, posisi, ukuran
3. **Mulut** (5 fitur): Lebar, tinggi, area, posisi, aspect ratio
4. **Region Atas** (3 fitur): Mean, std, kontras (area mata/alis)
5. **Region Tengah** (2 fitur): Mean, std (area hidung)
6. **Region Bawah** (3 fitur): Mean, std, kontras (area mulut)
7. **Rasio & Perbedaan** (3 fitur): Intensity ratio, contrast diff, variance
8. **Edge Detection** (2 fitur): Densitas edge, edge upper face
9. **Histogram** (2 fitur): Entropy, peak distribution
10. **Gradien** (2 fitur): Horizontal & vertical gradien

### Machine Learning Pipeline:

```
Input Image → Face Detection → Feature Extraction (35 fitur) 
    → Normalization (StandardScaler) 
    → Model Prediction (Random Forest / KNN)
    → Output: Emotion + Confidence + Explanation
```

### Classifier:

- **Primary**: Random Forest (n_estimators=300, max_depth=10)
- **Fallback**: K-Nearest Neighbors (k=3, distance-weighted)
- **Selection**: Automatic - pilih model dengan akurasi terbaik
- **Ensemble**: Voting classifier untuk hasil optimal

## 📊 Hasil Visualisasi

Aplikasi menampilkan:

| Komponen | Deskripsi |
|----------|-----------|
| **Emotion Badge** | Label emosi dengan confidence score (%) |
| **Annotated Image** | Gambar dengan bounding box wajah + keypoints berwarna |
| **Probability Bars** | Bar chart probabilitas untuk 3 kategori emosi |
| **Keypoints Detail** | Tabel koordinat (X, Y) untuk setiap titik mata & mulut |
| **Explanation** | Alasan AI mendeteksi emosi tersebut |

### Contoh Output:
```
Emosi: SENANG (86.5%)

Keypoints Terdeteksi:
• Mata #1 (biru): (245, 156) - Posisi mata menunjukkan karakteristik senyum
• Mata #2 (biru): (312, 158) - Mata menyipit (Duchenne smile)
• Mulut (merah): (278, 215) - Lebar mulut besar, bibir tersenyum

Probabilitas:
Senang:  86.5% ████████████████████
Sedih:   8.3%  ██
Marah:   5.2%  █

Alasan Deteksi:
✓ Mata: Mata menyipit (Duchenne smile)
✓ Mulut: Lebar mulut besar, bibir tersenyum lebar
✓ Region Bawah: Area mulut aktif dan terangkat
✓ Proporsi: Aktivasi otot di area pipi dan mulut
```

## 🎯 Tips untuk Hasil Terbaik

| ✅ DO | ❌ DON'T |
|-------|---------|
| Gunakan foto dengan pencahayaan baik | Foto gelap atau backlight |
| Wajah frontal dan jelas | Foto samping atau miring |
| Ekspresi yang jelas | Ekspresi ambiguous |
| Resolusi minimal 200x200px | Foto terlalu kecil/blur |
| Background sederhana | Background ramai/kompleks |
| Satu wajah per foto | Multiple faces |

## 📁 File-file Penting

| File | Deskripsi | Size |
|------|-----------|------|
| `train_model.py` | Script training model | ~12 KB |
| `app.py` | Flask web application | ~10 KB |
| `templates/index.html` | Web interface UI | ~15 KB |
| `requirements.txt` | Python dependencies | ~1 KB |
| `emotion_model.pkl` | Trained model | ~500 KB |
| `CARA_TINGKATKAN_AKURASI.md` | Panduan akurasi 80%+ | ~8 KB |

## 🛠️ Troubleshooting

### Problem: Wajah tidak terdeteksi
**Solusi**: 
- Pastikan wajah jelas, frontal, dan tidak tertutup
- Gunakan foto dengan resolusi minimal 200x200px
- Coba foto dengan pencahayaan lebih baik

### Problem: Akurasi rendah
**Solusi**: 
- Tambah data training (target: 200-300 gambar/kategori)
- Lihat [CARA_TINGKATKAN_AKURASI.md](CARA_TINGKATKAN_AKURASI.md)

### Problem: Error "Model belum di-load"
**Solusi**: 
- Jalankan `python train_model.py` terlebih dahulu
- Pastikan file `emotion_model.pkl` ada

### Problem: Error import cv2
**Solusi**: 
- Install ulang: `pip install opencv-python`
- Aktifkan virtual environment

### Problem: Port 5000 sudah digunakan
**Solusi**:
- Edit `app.py`, ubah port: `app.run(port=5001)`
- Atau stop aplikasi yang menggunakan port 5000

## 🔄 Update Model dengan Data Baru

```bash
# 1. Tambahkan gambar baru ke folder yang sesuai
#    data2/marah/, data2/sedih/, atau data2/senang/

# 2. Training ulang model
python train_model.py

# 3. Restart aplikasi Flask
# Tekan CTRL+C di terminal, lalu:
python app.py
```

## 📚 Teknologi yang Digunakan

| Kategori | Library/Tools | Versi |
|----------|---------------|-------|
| **Backend** | Flask | 2.3.3 |
| **Computer Vision** | OpenCV | 4.8.0 |
| **Machine Learning** | scikit-learn | 1.3.0 |
| **Numeric** | NumPy | 1.24.3 |
| **Image Processing** | Pillow | 10.0.0 |
| **Web Server** | Werkzeug | 2.3.7 |

## 📖 Dokumentasi Tambahan

- 📘 **[README.md](README.md)** - Dokumentasi utama (file ini)
- 📕 **[CARA_TINGKATKAN_AKURASI.md](CARA_TINGKATKAN_AKURASI.md)** - Panduan lengkap mencapai akurasi 80%+
- 📗 **[requirements.txt](requirements.txt)** - Daftar dependencies

## 🤝 Kontribusi

Untuk meningkatkan project ini:
1. Fork repository
2. Buat branch baru (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push ke branch (`git push origin feature/improvement`)
5. Buat Pull Request

## 📄 License

Project ini dibuat untuk keperluan pembelajaran dan riset.

---

**Catatan**: Akurasi 70% dengan dataset 317 gambar sudah cukup baik. Untuk akurasi 80%+, **wajib** menambah data training minimal 2-3x lipat. Lihat [CARA_TINGKATKAN_AKURASI.md](CARA_TINGKATKAN_AKURASI.md) untuk detail lengkap.

**Dibuat dengan ❤️ menggunakan Python, OpenCV, dan Machine Learning**
