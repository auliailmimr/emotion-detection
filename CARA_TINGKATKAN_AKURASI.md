# 🎯 Cara Meningkatkan Akurasi ke 80%+

Saat ini akurasi model: **70%** (dengan 317 gambar training)

## 📊 Mengapa Belum 80%+?

1. **Dataset Terlalu Kecil**: 317 gambar untuk 3 kelas = ~106 gambar/kelas
   - Minimum untuk 80%+: **200-300 gambar per kelas**
   - Ideal: **500-1000 gambar per kelas**

2. **Variasi Terbatas**: Dataset perlu lebih banyak variasi:
   - Berbagai pencahayaan
   - Berbagai sudut wajah
   - Berbagai usia dan gender
   - Berbagai intensitas emosi

## ✅ Solusi untuk Mencapai 80%+ Akurasi

### 1. **Tambah Data Training** (PALING PENTING!)

#### Opsi A: Download Dataset Publik
```bash
# FER2013 Dataset (Facial Expression Recognition)
# https://www.kaggle.com/datasets/msambare/fer2013

# CK+ Dataset (Extended Cohn-Kanade)
# http://www.consortium.ri.cmu.edu/ckagree/

# JAFFE Dataset (Japanese Female Facial Expression)
# https://zenodo.org/record/3451524
```

**Cara Menggunakan**:
1. Download dataset
2. Pisahkan gambar sesuai kategori (angry→marah, sad→sedih, happy→senang)
3. Copy ke folder `data2/marah/`, `data2/sedih/`, `data2/senang/`
4. Jalankan ulang `python train_model.py`

#### Opsi B: Data Augmentation (Perbanyak dari data yang ada)

Tambahkan kode ini di `train_model.py` sebelum training:

```python
from sklearn.utils import resample

# Data Augmentation
augmented_X = []
augmented_y = []

for label in np.unique(y):
    # Ambil data untuk kelas ini
    X_class = X[y == label]
    y_class = y[y == label]
    
    # Resample untuk mendapatkan 200 sampel per kelas
    X_resampled, y_resampled = resample(
        X_class, y_class,
        n_samples=200,
        random_state=42,
        replace=True  # Dengan replacement
    )
    
    augmented_X.append(X_resampled)
    augmented_y.append(y_resampled)

X = np.vstack(augmented_X)
y = np.hstack(augmented_y)
```

### 2. **Fine-Tuning Hyperparameter**

Untuk Random Forest yang lebih baik:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [8, 10, 12, 15],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2']
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_model, param_grid,
    cv=5,  # 5-fold cross validation
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
```

### 3. **Gunakan Deep Learning** (Untuk 85%+ akurasi)

Install library tambahan:
```bash
pip install tensorflow keras
```

Buat model CNN:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 4. **Cross-Validation untuk Evaluasi Lebih Akurat**

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross validation
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Std deviation: {cv_scores.std()*100:.2f}%")
```

### 5. **Feature Engineering Lebih Lanjut**

Tambahkan fitur:
- LBP (Local Binary Patterns)
- HOG (Histogram of Oriented Gradients)
- Facial landmarks (68 points) dengan dlib
- Geometric features (jarak antar landmarks)

## 📈 Ekspektasi Akurasi Berdasarkan Dataset Size

| Jumlah Gambar/Kelas | Ekspektasi Akurasi | Model |
|---------------------|-------------------|-------|
| 100-150 | 60-70% | KNN/RF |
| 150-300 | 70-80% | RF/SVM |
| 300-500 | 80-85% | RF/Ensemble |
| 500-1000 | 85-90% | Deep Learning |
| 1000+ | 90-95% | CNN/ResNet |

## 🎯 Rekomendasi Step-by-Step

### Untuk Mencapai 80% (Realistis):

1. **Download FER2013 atau CK+ dataset**
2. **Tambahkan minimal 200 gambar per kategori**
3. **Gunakan Random Forest dengan GridSearchCV**
4. **Cross-validation untuk evaluasi**
5. **Test dengan data real-world**

### Untuk Mencapai 85%+ (Advanced):

1. **Gunakan 500+ gambar per kategori**
2. **Implementasi CNN dengan transfer learning (VGG16/ResNet)**
3. **Data augmentation (rotation, flip, brightness)**
4. **Ensemble multiple deep learning models**

## 🔍 Monitoring & Debugging

Tambahkan confusion matrix untuk melihat kesalahan prediksi:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## 💡 Tips Praktis

1. ✅ **Kualitas > Kuantitas**: Pastikan gambar training berkualitas baik
2. ✅ **Balance Dataset**: Jumlah gambar per kelas harus seimbang
3. ✅ **Validasi Silang**: Gunakan cross-validation, bukan hanya single test split
4. ✅ **Clean Data**: Hapus gambar yang ambiguous atau blur
5. ✅ **Diverse Data**: Variasi umur, gender, etnisitas, pencahayaan

## 📚 Resource Tambahan

- **Dataset**: 
  - FER2013: https://www.kaggle.com/datasets/msambare/fer2013
  - CK+: http://www.consortium.ri.cmu.edu/ckagree/
  - JAFFE: https://zenodo.org/record/3451524
  
- **Tutorial**:
  - Facial Expression Recognition dengan CNN
  - Transfer Learning untuk Emotion Detection
  - Data Augmentation Techniques

---

**Catatan**: Dengan dataset saat ini (317 gambar), akurasi 70% sudah cukup bagus. Untuk 80%+, WAJIB menambah data training minimal 2-3x lipat.
