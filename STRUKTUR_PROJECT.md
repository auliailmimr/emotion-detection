# 📁 Struktur Project - Emotion Detection System

## Overview
```
emotion-detection/
│
├── 📄 README.md                          # Dokumentasi utama project
├── 📄 CARA_TINGKATKAN_AKURASI.md        # Panduan meningkatkan akurasi ke 80%+
├── 📄 requirements.txt                   # Daftar dependencies Python
├── 📄 .gitignore                         # File yang diabaikan oleh Git
│
├── 🐍 train_model.py                     # Script untuk training model ML
├── 🌐 app.py                             # Flask web application (backend)
├── 📦 emotion_model.pkl                 # Trained model (generated)
│
├── 📂 templates/                         # Folder HTML templates
│   └── index.html                        # Web interface (frontend)
│
├── 📂 data2/                             # Dataset training
│   ├── 📂 marah/                         # 108 gambar ekspresi marah
│   ├── 📂 sedih/                         # 109 gambar ekspresi sedih
│   ├── 📂 senang/                        # 100 gambar ekspresi senang
│   ├── haarcascade_frontalface_default.xml  # Haar cascade untuk deteksi wajah
│   ├── haarcascade_eye.xml               # Haar cascade untuk deteksi mata
│   └── haarcascade_mcs_mouth.xml         # Haar cascade untuk deteksi mulut
│
├── 📂 uploads/                           # Folder untuk gambar yang diupload
│   └── .gitkeep                          # Keep folder dalam git
│
└── 📂 .venv/                             # Virtual environment Python
    └── (dependencies Python disimpan di sini)
```

## Detail File Utama

### 1. train_model.py
**Purpose**: Training machine learning model
**Features**:
- Ekstraksi 35 fitur dari gambar wajah
- Training Random Forest & KNN
- Ensemble voting untuk akurasi optimal
- Normalisasi dengan StandardScaler
- Evaluasi dengan classification report

**Output**: `emotion_model.pkl`

**Usage**: 
```bash
python train_model.py
```

---

### 2. app.py
**Purpose**: Flask web application (backend)
**Features**:
- REST API endpoint `/predict` untuk prediksi
- File upload handling
- Ekstraksi fitur real-time
- Visualisasi dengan keypoints
- JSON response dengan error handling

**Routes**:
- `GET /` - Halaman utama
- `POST /predict` - Prediksi emosi dari gambar
- `GET /uploads/<filename>` - Serve uploaded images

**Usage**:
```bash
python app.py
# Access: http://localhost:5000
```

---

### 3. templates/index.html
**Purpose**: Web interface (frontend)
**Features**:
- Drag & drop file upload
- Image preview
- Progress indicator
- Result visualization dengan:
  - Annotated image dengan keypoints
  - Emotion badge + confidence score
  - Probability bars
  - Keypoints coordinates table
  - Explanation text
- Responsive design
- Error handling

**Technology**: HTML5, CSS3, JavaScript (Vanilla)

---

### 4. data2/
**Purpose**: Dataset untuk training
**Structure**:
```
data2/
├── marah/         # Angry emotion images
├── sedih/         # Sad emotion images
└── senang/        # Happy emotion images
```

**Format**: JPG, PNG
**Current Size**: 317 images total
**Recommended Size**: 600-900 images (200-300 per kategori)

---

### 5. requirements.txt
**Purpose**: Python dependencies
**Contents**:
```
flask==2.3.3
opencv-python==4.8.0.76
numpy==1.24.3
scikit-learn==1.3.0
pillow==10.0.0
werkzeug==2.3.7
```

---

### 6. emotion_model.pkl
**Purpose**: Trained machine learning model
**Contains**:
- Model object (RandomForest/KNN)
- StandardScaler object
- Emotion labels mapping
- Feature dimensions

**Size**: ~500 KB
**Generated**: After running `train_model.py`

---

## File yang Diabaikan (.gitignore)

```
__pycache__/           # Python cache
*.pyc                  # Compiled Python
*.pkl                  # Model files
venv/                  # Virtual environment
uploads/*              # Uploaded images
.vscode/               # VS Code settings
.ipynb_checkpoints     # Jupyter checkpoints
*.log                  # Log files
```

---

## Workflow

```
1. SETUP
   ├── Install dependencies (pip install -r requirements.txt)
   └── Prepare dataset (add images to data2/)

2. TRAINING
   ├── Run: python train_model.py
   ├── Extract 35 features per image
   ├── Train models (RandomForest + KNN)
   └── Save: emotion_model.pkl

3. DEPLOYMENT
   ├── Run: python app.py
   └── Access: http://localhost:5000

4. USAGE
   ├── Upload image via web interface
   ├── Server extracts features
   ├── Model predicts emotion
   └── Display results with visualization
```

---

## File Size Guide

| File/Folder | Size | Note |
|-------------|------|------|
| `train_model.py` | ~12 KB | Python script |
| `app.py` | ~10 KB | Python script |
| `templates/index.html` | ~15 KB | HTML/CSS/JS |
| `requirements.txt` | ~150 B | Text file |
| `emotion_model.pkl` | ~500 KB | Binary file (generated) |
| `data2/` | ~50 MB | 317 images |
| `.venv/` | ~500 MB | Python packages |
| **TOTAL** | ~550 MB | Full project |

---

## Clean Project (for sharing)

Jika ingin share project tanpa file besar:

```bash
# Hapus file yang di-generate
rm emotion_model.pkl

# Hapus uploads
rm -rf uploads/*

# Hapus virtual environment
rm -rf .venv/

# Ukuran final: ~50 MB (hanya source code + dataset)
```

---

## Dependencies Tree

```
emotion-detection
├── Flask (web framework)
│   └── Werkzeug (WSGI utilities)
├── OpenCV (computer vision)
│   └── NumPy (numerical operations)
├── scikit-learn (machine learning)
│   └── NumPy
└── Pillow (image processing)
```

---

**Last Updated**: November 9, 2025  
**Project Version**: 1.0  
**Python Version**: 3.13.9
