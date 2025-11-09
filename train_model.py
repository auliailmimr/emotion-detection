import cv2
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def extract_facial_features(image_path):
    """
    Ekstraksi fitur wajah menggunakan Haar Cascade untuk deteksi
    dan perhitungan proporsi bagian wajah
    """
    # Load cascade classifiers
    face_cascade = cv2.CascadeClassifier('data2/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data2/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('data2/haarcascade_mcs_mouth.xml')
    
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah dengan parameter yang seimbang
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
    
    if len(faces) == 0:
        # Coba lagi dengan parameter lebih longgar
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    
    # Ambil wajah pertama yang terdeteksi
    (fx, fy, fw, fh) = faces[0]
    face_roi = gray[fy:fy+fh, fx:fx+fw]
    
    # Deteksi mata dalam ROI wajah dengan parameter seimbang
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))
    
    # Jika tidak ada mata terdeteksi, coba parameter lebih longgar
    if len(eyes) == 0:
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
    
    # Deteksi mulut dalam ROI wajah
    mouth_roi = face_roi[int(fh*0.5):fh, 0:fw]  # Bagian bawah wajah
    mouths = mouth_cascade.detectMultiScale(mouth_roi, scaleFactor=1.4, minNeighbors=12, minSize=(25, 15))
    
    # Jika tidak ada mulut terdeteksi, coba parameter lebih longgar
    if len(mouths) == 0:
        mouths = mouth_cascade.detectMultiScale(mouth_roi, scaleFactor=1.5, minNeighbors=8, minSize=(20, 12))
    
    # Ekstraksi fitur yang lebih diskriminatif
    features = []
    
    # Fitur 1-4: Koordinat wajah (normalisasi)
    features.extend([fx/image.shape[1], fy/image.shape[0], fw/image.shape[1], fh/image.shape[0]])
    
    # Fitur 5-8: Mata - posisi dan ukuran
    num_eyes = min(len(eyes), 2)
    features.append(num_eyes)
    
    if len(eyes) >= 2:
        # Urutkan mata dari kiri ke kanan
        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        (ex1, ey1, ew1, eh1) = eyes_sorted[0]  # Mata kiri
        (ex2, ey2, ew2, eh2) = eyes_sorted[1]  # Mata kanan
        
        # Jarak antar mata (penting untuk ekspresi)
        eye_distance = abs(ex2 - ex1) / fw
        # Posisi vertikal rata-rata mata
        avg_eye_y = ((ey1 + ey2) / 2) / fh
        # Ukuran mata rata-rata
        avg_eye_size = ((ew1 * eh1 + ew2 * eh2) / 2) / (fw * fh)
        features.extend([eye_distance, avg_eye_y, avg_eye_size])
    elif len(eyes) == 1:
        (ex, ey, ew, eh) = eyes[0]
        features.extend([0.3, ey/fh, (ew*eh)/(fw*fh)])
    else:
        features.extend([0.3, 0.3, 0.05])
    
    # Fitur 9-12: Mulut - ukuran dan posisi (sangat penting untuk emosi)
    if len(mouths) > 0:
        (mx, my, mw, mh) = mouths[0]
        # Lebar mulut relatif (senyum = lebar, sedih = sempit)
        mouth_width_ratio = mw / fw
        # Tinggi mulut (terbuka vs tertutup)
        mouth_height_ratio = mh / fh
        # Area mulut
        mouth_area = (mw * mh) / (fw * fh)
        # Posisi vertikal mulut
        mouth_y = (my + int(fh*0.5)) / fh
        features.extend([mouth_width_ratio, mouth_height_ratio, mouth_area, mouth_y])
        
        # Fitur tambahan: Rasio lebar vs tinggi mulut (penting!)
        mouth_aspect_ratio = mw / (mh + 1e-6)  # Senyum lebar = ratio tinggi
        features.append(mouth_aspect_ratio)
    else:
        features.extend([0.3, 0.1, 0.03, 0.75, 3.0])  # Default dengan ratio normal
    
    # Fitur 13-16: Analisis region wajah atas (mata dan alis)
    upper_face = face_roi[0:int(fh*0.45), :]
    upper_mean = np.mean(upper_face) / 255.0
    upper_std = np.std(upper_face) / 255.0
    # Deteksi kontras di region atas (alis terangkat/turun)
    upper_contrast = np.max(upper_face) - np.min(upper_face)
    upper_contrast_norm = upper_contrast / 255.0
    features.extend([upper_mean, upper_std, upper_contrast_norm])
    
    # Fitur 17-20: Analisis region wajah tengah (hidung)
    middle_face = face_roi[int(fh*0.35):int(fh*0.65), :]
    middle_mean = np.mean(middle_face) / 255.0
    middle_std = np.std(middle_face) / 255.0
    features.extend([middle_mean, middle_std])
    
    # Fitur 21-25: Analisis region wajah bawah (mulut dan dagu)
    lower_face = face_roi[int(fh*0.55):, :]
    lower_mean = np.mean(lower_face) / 255.0
    lower_std = np.std(lower_face) / 255.0
    lower_contrast = np.max(lower_face) - np.min(lower_face)
    lower_contrast_norm = lower_contrast / 255.0
    features.extend([lower_mean, lower_std, lower_contrast_norm])
    
    # Fitur 26-28: Rasio dan perbedaan antar region (indikator kuat untuk ekspresi)
    # Rasio upper vs lower (marah = upper tegang, senang = lower aktif)
    intensity_ratio = upper_mean / (lower_mean + 1e-6)
    # Perbedaan kontras (emosi kuat = kontras tinggi)
    contrast_diff = abs(upper_contrast_norm - lower_contrast_norm)
    # Variasi keseluruhan
    face_variance = np.var(face_roi) / (255.0 * 255.0)
    features.extend([intensity_ratio, contrast_diff, face_variance])
    
    # FITUR TAMBAHAN untuk akurasi lebih tinggi (29-35)
    
    # Fitur 30-31: Edge detection (deteksi kontur wajah untuk ekspresi)
    edges = cv2.Canny(face_roi, 50, 150)
    edge_density = np.sum(edges > 0) / (fw * fh)
    edge_upper = np.sum(edges[0:int(fh*0.5), :] > 0) / (fw * int(fh*0.5))
    features.extend([edge_density, edge_upper])
    
    # Fitur 32-33: Histogram analisis (distribusi intensitas)
    hist = cv2.calcHist([face_roi], [0], None, [16], [0, 256])
    hist_entropy = -np.sum((hist/np.sum(hist)) * np.log2((hist/np.sum(hist)) + 1e-10))
    hist_peak = np.argmax(hist) / 16.0  # Normalize
    features.extend([hist_entropy, hist_peak])
    
    # Fitur 34-35: Gradien horizontal dan vertikal (perubahan intensitas)
    grad_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
    grad_x_mean = np.mean(np.abs(grad_x)) / 255.0
    grad_y_mean = np.mean(np.abs(grad_y)) / 255.0
    features.extend([grad_x_mean, grad_y_mean])
    
    return np.array(features), (fx, fy, fw, fh), eyes, mouths

def load_dataset(data_dir):
    """
    Load dataset dari folder marah, sedih, senang
    """
    emotions = {
        'marah': 0,
        'sedih': 1,
        'senang': 2
    }
    
    X = []
    y = []
    image_paths = []
    
    for emotion, label in emotions.items():
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Folder {emotion_dir} tidak ditemukan!")
            continue
        
        files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing {emotion}: {len(files)} images...")
        
        for filename in files:
            filepath = os.path.join(emotion_dir, filename)
            features = extract_facial_features(filepath)
            
            if features is not None:
                X.append(features[0])
                y.append(label)
                image_paths.append(filepath)
            else:
                print(f"  Gagal ekstraksi fitur: {filename}")
    
    return np.array(X), np.array(y), image_paths, emotions

def train_emotion_detector():
    """
    Train KNN classifier untuk emotion detection
    """
    print("="*60)
    print("EMOTION DETECTION - TRAINING MODEL")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    X, y, image_paths, emotion_labels = load_dataset('data2')
    
    print(f"\nTotal sampel berhasil diekstrak: {len(X)}")
    print(f"Distribusi kelas:")
    for emotion, label in sorted(emotion_labels.items(), key=lambda x: x[1]):
        count = np.sum(y == label)
        print(f"  {emotion.capitalize()}: {count} images")
    
    if len(X) < 10:
        print("\nPeringatan: Dataset terlalu kecil! Minimal 10 sampel diperlukan.")
        return None
    
    # Split dataset
    print("\n2. Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Data training: {len(X_train)} samples")
    print(f"Data testing: {len(X_test)} samples")
    
    # Normalisasi fitur dengan StandardScaler
    print("\n3. Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Multiple Models dan pilih yang terbaik
    print("\n4. Training Multiple Classifiers...")
    
    # Random Forest - Tuned untuk data kecil
    print("   - Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300,          # Lebih banyak trees
        max_depth=10,              # Lebih shallow untuk prevent overfitting
        min_samples_split=3,       # Lebih kecil
        min_samples_leaf=1,        # Allow lebih detail
        max_features='log2',       # Feature selection
        bootstrap=True,
        oob_score=True,            # Out-of-bag score
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # KNN dengan parameter optimal
    print("   - Training KNN...")
    knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
    knn_model.fit(X_train_scaled, y_train)
    knn_pred = knn_model.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    
    # Ensemble Voting - Combine both models
    print("   - Creating Ensemble Voting...")
    from sklearn.ensemble import VotingClassifier
    ensemble = VotingClassifier(
        estimators=[('rf', rf_model), ('knn', knn_model)],
        voting='soft',  # Use probability voting
        weights=[1.5, 1]  # RF slightly higher weight
    )
    ensemble.fit(X_train_scaled, y_train)
    ensemble_pred = ensemble.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"\n   Random Forest Accuracy: {rf_accuracy*100:.2f}%")
    print(f"   KNN Accuracy: {knn_accuracy*100:.2f}%")
    print(f"   Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")
    
    # Pilih model terbaik
    accuracies = {'RandomForest': (rf_model, rf_pred, rf_accuracy), 
                  'KNN': (knn_model, knn_pred, knn_accuracy),
                  'Ensemble': (ensemble, ensemble_pred, ensemble_accuracy)}
    
    best_name = max(accuracies, key=lambda k: accuracies[k][2])
    best_model, y_pred, accuracy = accuracies[best_name]
    model_name = best_name
    
    print(f"\n   ✓ Using {model_name} (Best accuracy: {accuracy*100:.2f}%)")
    
    # Evaluate
    print("\n5. Evaluating model...")
    
    print(f"\nAkurasi Model: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    emotion_names = {v: k for k, v in emotion_labels.items()}
    # Only include classes that are actually present in the data
    unique_labels = sorted(np.unique(y))
    target_names = [emotion_names[i].capitalize() for i in unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))
    
    # Save model
    print("\n6. Saving model...")
    model_data = {
        'model': best_model,
        'model_name': model_name,
        'scaler': scaler,
        'emotion_labels': emotion_labels,
        'feature_dim': X.shape[1]
    }
    
    with open('emotion_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model ({model_name}) berhasil disimpan ke 'emotion_model.pkl'")
    print("\n" + "="*60)
    
    return best_model, emotion_labels

if __name__ == "__main__":
    train_emotion_detector()
