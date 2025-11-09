from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Buat folder uploads jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
try:
    with open('emotion_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    knn_model = model_data['model']
    scaler = model_data.get('scaler', None)
    emotion_labels = model_data['emotion_labels']
    # Reverse mapping: label -> emotion name
    label_to_emotion = {v: k for k, v in emotion_labels.items()}
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    knn_model = None
    scaler = None
    emotion_labels = None
    label_to_emotion = None

def extract_facial_features(image_path):
    """
    Ekstraksi fitur wajah - sama seperti di training
    """
    # Load cascade classifiers
    face_cascade = cv2.CascadeClassifier('data2/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data2/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('data2/haarcascade_mcs_mouth.xml')
    
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah dengan parameter yang seimbang
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
    
    if len(faces) == 0:
        # Coba lagi dengan parameter lebih longgar
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None, None, None, None
    
    # Ambil wajah pertama yang terdeteksi
    (fx, fy, fw, fh) = faces[0]
    face_roi = gray[fy:fy+fh, fx:fx+fw]
    
    # Deteksi mata dalam ROI wajah dengan parameter seimbang
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))
    
    # Jika tidak ada mata terdeteksi, coba parameter lebih longgar
    if len(eyes) == 0:
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
    
    # Deteksi mulut dalam ROI wajah
    mouth_roi = face_roi[int(fh*0.5):fh, 0:fw]
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
    
    return np.array(features), image, (fx, fy, fw, fh), eyes, mouths

def create_visualization(image, face_coords, eyes, mouths, emotion, confidence):
    """
    Membuat visualisasi dengan bounding box dan keypoints
    """
    vis_image = image.copy()
    fx, fy, fw, fh = face_coords
    
    # Gambar kotak wajah
    cv2.rectangle(vis_image, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
    
    # Gambar mata
    keypoints = []
    for (ex, ey, ew, eh) in eyes:
        eye_x = fx + ex + ew//2
        eye_y = fy + ey + eh//2
        cv2.circle(vis_image, (eye_x, eye_y), 5, (255, 0, 0), -1)
        cv2.rectangle(vis_image, (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (255, 0, 0), 2)
        keypoints.append({
            'name': 'Mata',
            'x': eye_x,
            'y': eye_y,
            'color': 'biru',
            'description': 'Posisi dan ukuran mata mempengaruhi deteksi ekspresi'
        })
    
    # Gambar mulut
    for (mx, my, mw, mh) in mouths:
        mouth_x = fx + mx + mw//2
        mouth_y = fy + int(fh*0.5) + my + mh//2
        cv2.circle(vis_image, (mouth_x, mouth_y), 5, (0, 0, 255), -1)
        cv2.rectangle(vis_image, (fx+mx, fy+int(fh*0.5)+my), 
                     (fx+mx+mw, fy+int(fh*0.5)+my+mh), (0, 0, 255), 2)
        keypoints.append({
            'name': 'Mulut',
            'x': mouth_x,
            'y': mouth_y,
            'color': 'merah',
            'description': f'Lebar mulut: {mw}px, Tinggi: {mh}px - Bentuk mulut penting untuk ekspresi'
        })
    
    # Tambah label emosi
    label_text = f"{emotion.upper()} ({confidence:.1f}%)"
    cv2.putText(vis_image, label_text, (fx, fy-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return vis_image, keypoints

def get_emotion_explanation(emotion, features, eyes, mouths):
    """
    Memberikan penjelasan mengapa emosi terdeteksi
    """
    explanations = []
    
    if emotion == 'marah':
        explanations.append("Ekspresi MARAH terdeteksi karena:")
        if len(eyes) >= 2:
            explanations.append("• Mata: Posisi dan jarak mata menunjukkan ketegangan")
        if len(mouths) > 0:
            explanations.append("• Mulut: Bentuk mulut tegang/tertutup rapat")
        explanations.append("• Region Atas: Kontras tinggi menunjukkan alis mengerut")
        explanations.append("• Tekstur: Kontraksi otot wajah yang tegang")
        
    elif emotion == 'sedih':
        explanations.append("Ekspresi SEDIH terdeteksi karena:")
        if len(eyes) > 0:
            explanations.append("• Mata: Posisi mata menunjukkan kelopak menurun")
        if len(mouths) > 0:
            explanations.append("• Mulut: Sudut bibir cenderung menurun")
        explanations.append("• Region Bawah: Area mulut menunjukkan relaksasi/kesedihan")
        explanations.append("• Proporsi: Keseimbangan wajah menunjukkan ekspresi pasif")
        
    elif emotion == 'senang':
        explanations.append("Ekspresi SENANG terdeteksi karena:")
        if len(eyes) > 0:
            explanations.append("• Mata: Mata menyipit (Duchenne smile)")
        if len(mouths) > 0:
            explanations.append("• Mulut: Lebar mulut besar, bibir tersenyum lebar")
        explanations.append("• Region Bawah: Area mulut aktif dan terangkat")
        explanations.append("• Proporsi: Aktivasi otot di area pipi dan mulut")
    
    return explanations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if knn_model is None:
            return jsonify({'error': 'Model belum di-load. Jalankan train_model.py terlebih dahulu.'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Simpan file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Ekstraksi fitur
            features, image, face_coords, eyes, mouths = extract_facial_features(filepath)
            
            if features is None:
                return jsonify({'error': 'Wajah tidak terdeteksi dalam gambar'}), 400
            
            # Normalisasi fitur jika scaler tersedia
            features_to_predict = features.reshape(1, -1)
            if scaler is not None:
                features_to_predict = scaler.transform(features_to_predict)
            
            # Prediksi
            prediction = knn_model.predict(features_to_predict)[0]
            probabilities = knn_model.predict_proba(features_to_predict)[0]
            confidence = probabilities[prediction] * 100
            
            emotion = label_to_emotion[prediction]
            
            # Buat visualisasi
            vis_image, keypoints = create_visualization(image, face_coords, eyes, mouths, emotion, confidence)
            
            # Simpan gambar hasil
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, vis_image)
            
            # Get explanation
            explanations = get_emotion_explanation(emotion, features, eyes, mouths)
            
            # Convert ke base64 untuk dikirim ke frontend
            _, buffer = cv2.imencode('.jpg', vis_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Convert numpy types to Python native types for JSON serialization
            keypoints_serializable = []
            for kp in keypoints:
                keypoints_serializable.append({
                    'name': str(kp['name']),
                    'x': int(kp['x']),
                    'y': int(kp['y']),
                    'color': str(kp['color']),
                    'description': str(kp['description'])
                })
            
            return jsonify({
                'emotion': emotion.capitalize(),
                'confidence': float(round(confidence, 2)),
                'image': img_base64,
                'keypoints': keypoints_serializable,
                'explanations': explanations,
                'all_probabilities': {
                    label_to_emotion[i].capitalize(): float(round(prob*100, 2)) 
                    for i, prob in enumerate(probabilities)
                }
            })
    
    except Exception as e:
        # Log error dan kirim response JSON
        print(f"Error in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
