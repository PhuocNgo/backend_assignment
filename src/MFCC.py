import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample

target_sample_rate = 44100


def extract_features(audio_file, n_fft=128, hop_length=32, n_mfcc=13):
    try:
        y, sr = sf.read(audio_file)

        # Nếu là stereo, chuyển thành mono
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Chuẩn hóa sample rate
        if sr != target_sample_rate:
            y = resample(y, int(len(y) * target_sample_rate / sr))
            sr = target_sample_rate

        # Lấy đúng 5 giây đầu
        samples_5_sec = target_sample_rate * 5
        y_5_sec = y[:samples_5_sec]

        # Nếu file ngắn hơn 5s, pad 0
        if len(y_5_sec) < samples_5_sec:
            y_5_sec = np.pad(y_5_sec, (0, samples_5_sec - len(y_5_sec)))

        # Trích xuất MFCC
        mfccs = librosa.feature.mfcc(
            y=y_5_sec, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # Chuẩn hóa từng hệ số MFCC theo thời gian
        mfccs_normalized = (mfccs - np.mean(mfccs, axis=1, keepdims=True)
                            ) / (np.std(mfccs, axis=1, keepdims=True) + 1e-6)

        # Lấy trung bình mỗi hệ số → vector 13 chiều
        mean_mfcc = np.mean(mfccs_normalized, axis=1)

        return mean_mfcc  # Vector 13 chiều

    except Exception as e:
        print(f"Lỗi khi xử lý file {audio_file}: {e}")
        return None
