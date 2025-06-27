import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
import readFile
import re

# Tắt thông báo oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Tiền xử lý âm thanh và trích xuất Mel-spectrogram


def preprocess_audio(file_path, duration=5, sr=44100, n_mels=128):
    try:
        y, sr_orig = librosa.load(file_path, sr=sr, duration=duration)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < duration * sr:
            y = np.pad(y, (0, duration * sr - len(y)))
        else:
            y = y[:duration * sr]
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
        mel = librosa.power_to_db(mel, ref=np.max)
        return mel[..., np.newaxis]  # Shape: (n_mels, time_steps, 1)
    except Exception as e:
        print(f"Lỗi khi xử lý file {file_path}: {e}")
        return None

# Tính pitch trung bình


def calculate_pitch_mean(file_path, sr=44100):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > 0]
        return np.mean(pitch_values) if pitch_values.size > 0 else 120.0
    except:
        return 120.0

# Xây dựng mô hình CNN


def build_cnn_model(input_shape=(128, 431, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Sửa cảnh báo input_shape
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),  # Vector đặc trưng 128 chiều
        layers.Lambda(lambda x: tf.math.l2_normalize(
            x, axis=1))  # Chuẩn hóa L2
    ])
    return model

# Lưu trữ đặc trưng vào MySQL


def store_features(data_dir, model, db_config):
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_features (
            id INT AUTO_INCREMENT PRIMARY KEY,
            path VARCHAR(255) NOT NULL,
            feature BLOB,
            pitch_mean FLOAT,
            gender_label VARCHAR(50),
            range_label VARCHAR(50)
        )
    ''')

    audio_files = readFile.list_wav_files(data_dir)
    for audio in audio_files:
        audio_path = os.path.join(data_dir, audio)
        mel = preprocess_audio(audio_path)
        if mel is None:
            continue
        mel = mel[np.newaxis, ...]
        feature = model.predict(mel, verbose=0)[0]
        pitch_mean = calculate_pitch_mean(audio_path)

        # Trích xuất nhãn từ tên file
        match = re.match(r'(\d+-\d+|\d+\+)_([A-Za-z]+)_\d+\.wav', audio)
        range_label = match.group(1) if match else 'Unknown'
        gender_label = match.group(2) if match else 'Unknown'

        cursor.execute('''
            INSERT INTO voice_features (path, feature, pitch_mean, gender_label, range_label)
            VALUES (%s, %s, %s, %s, %s)
        ''', (audio_path, feature.tobytes(), pitch_mean, gender_label, range_label))

    conn.commit()
    conn.close()

# Tìm kiếm giọng nói đàn ông


def search_male_voice(input_file, model, db_config, top_k=100):
    mel = preprocess_audio(input_file)
    if mel is None:
        return [], []
    mel = mel[np.newaxis, ...]
    input_feature = model.predict(mel, verbose=0)[0]

    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT path, feature, pitch_mean, gender_label FROM voice_features")
    rows = cursor.fetchall()
    conn.close()

    male_paths, male_features = [], []
    for row in rows:
        path, feature_blob, pitch_mean, gender_label = row
        if gender_label.lower() == 'nam' or pitch_mean < 180:
            male_paths.append(path)
            male_features.append(np.frombuffer(feature_blob, dtype=np.float32))

    if not male_features:
        print("Không tìm thấy giọng nói đàn ông trong CSDL.")
        return [], []

    similarities = cosine_similarity([input_feature], male_features)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_files = [male_paths[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]
    return top_files, top_scores  # Trả về hai danh sách riêng biệt

# Demo hệ thống


def main():
    data_dir = "D:/__pycache__/wav"
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'fuokqo160802',
        'database': 'csdl_dpt'
    }

    # Xây dựng mô hình
    model = build_cnn_model()

    # Lưu trữ đặc trưng
    store_features(data_dir, model, db_config)

    # Demo với file đầu vào
    input_file = "C:/Users/Admin/Downloads/Yêu-cầu-18_06_2025.wav"
    top_files, top_scores = search_male_voice(input_file, model, db_config)

    for i, (file, score) in enumerate(zip(top_files, top_scores)):
        print(f"Top {i+1}: File {file} với độ tương đồng {score:.4f}")


if __name__ == "__main__":
    main()
