# 🎯 QUICK REFERENCE - Emotion Detection System

## 📦 Instalasi & Setup (5 menit)

```bash
# 1. Masuk ke folder project
cd emotion-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Training model
python train_model.py

# 4. Jalankan aplikasi
python app.py

# 5. Buka browser
# http://localhost:5000
```

---

## 🚀 Command Cheat Sheet

### Training Model
```bash
# Basic training
python train_model.py

# Output: emotion_model.pkl (Model file)
```

### Jalankan Aplikasi
```bash
# Start Flask server
python app.py

# Akses: http://localhost:5000
# Port default: 5000
```

### Install/Update Dependencies
```bash
# Install semua
pip install -r requirements.txt

# Install individual
pip install flask opencv-python scikit-learn numpy pillow
```

---

## 📊 Status Project

| Item | Status | Detail |
|------|--------|--------|
| **Dataset** | ✅ 317 images | Marah: 108, Sedih: 109, Senang: 100 |
| **Model** | ✅ Trained | RandomForest + KNN Ensemble |
| **Akurasi** | ✅ 70.31% | Good untuk dataset size |
| **Fitur** | ✅ 35 features | Mata, mulut, tekstur, edge, dll |
| **Web App** | ✅ Running | Flask @ port 5000 |
| **Error Handling** | ✅ Fixed | JSON error sudah resolved |

---

## 🎯 Performance Metrics

```
Overall Accuracy: 70.31%

Per-Class Performance:
├── Marah  : Precision 67%, Recall 55% (F1: 60%)
├── Sedih  : Precision 58%, Recall 64% (F1: 61%)
└── Senang : Precision 86%, Recall 95% (F1: 90%) ⭐
```

---

## 📂 File Structure (Simplified)

```
emotion-detection/
├── 📄 train_model.py      # Training script
├── 📄 app.py              # Web app
├── 📄 emotion_model.pkl   # Model file
├── 📂 templates/          # HTML files
├── 📂 data2/              # Dataset
└── 📂 uploads/            # Uploaded images
```

---

## 🔧 Troubleshooting Quick Fix

| Problem | Quick Fix |
|---------|-----------|
| Wajah tidak terdeteksi | Gunakan foto frontal, pencahayaan baik |
| Error cv2 | `pip install opencv-python` |
| Port 5000 used | Edit app.py: `app.run(port=5001)` |
| Model not found | Run `python train_model.py` |
| Low accuracy | Add more training data (200+/class) |

---

## 📖 Dokumentasi Lengkap

| File | Isi |
|------|-----|
| **README.md** | Dokumentasi utama, setup, usage |
| **CARA_TINGKATKAN_AKURASI.md** | Panduan akurasi 80%+ |
| **STRUKTUR_PROJECT.md** | Detail struktur file & folder |

---

## 🎨 Features Highlight

✅ **3 Emosi**: Marah, Sedih, Senang  
✅ **35 Fitur AI**: Computer vision + ML  
✅ **Web Interface**: Drag & drop upload  
✅ **Real-time**: Instant prediction  
✅ **Visualisasi**: Keypoints + confidence  
✅ **Explanation**: AI reasoning  

---

## 💡 Tips Cepat

1. **Foto bagus** = Hasil akurat
   - Frontal, jelas, pencahayaan baik
   
2. **Tambah data** = Akurasi naik
   - Target: 200-300 gambar/kategori
   
3. **Re-train** setelah tambah data
   - `python train_model.py`
   
4. **Restart app** setelah training
   - CTRL+C → `python app.py`

---

## 🎯 Next Steps (Untuk Akurasi 80%+)

**Priority 1**: Tambah Data
- Download FER2013 dataset
- Atau ambil dari CK+ / JAFFE
- Target: 600-900 images total

**Priority 2**: Fine-tuning
- GridSearchCV untuk optimal parameters
- Cross-validation 5-fold
- Data augmentation

**Priority 3**: Deep Learning
- CNN dengan TensorFlow/Keras
- Transfer learning (VGG16/ResNet)
- Akurasi target: 85-90%

---

## 📞 Support

**Dokumentasi**:
- [README.md](README.md) - Setup & usage
- [CARA_TINGKATKAN_AKURASI.md](CARA_TINGKATKAN_AKURASI.md) - Advanced
- [STRUKTUR_PROJECT.md](STRUKTUR_PROJECT.md) - Architecture

**Check List**:
- ✅ Dependencies installed
- ✅ Dataset ready (data2/ folder)
- ✅ Model trained (emotion_model.pkl exists)
- ✅ Flask running (port 5000)
- ✅ No errors in terminal

---

**Version**: 1.0  
**Last Updated**: November 9, 2025  
**Status**: ✅ Production Ready
