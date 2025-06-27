from pydub import AudioSegment
import numpy as np
import librosa

def funcPitch_pydub(path):
    try:
        audio = AudioSegment.from_wav(path)
    except Exception as e:
        print(f"Lỗi khi mở file {path}: {e}")
        return [0] * 7

    # Chuyển đổi thành mono và lấy tín hiệu âm thanh
    audio = audio.set_channels(1)  # Đảm bảo là mono
    audio = audio.set_frame_rate(44100)  # Đảm bảo tần số mẫu là 44100
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # Chia tín hiệu thành các đoạn nhỏ
    num_segments = 7
    segment_length = len(samples) // num_segments

    result = []
    for i in range(num_segments):
        segment = samples[i * segment_length : (i + 1) * segment_length]
        # Trích xuất tần số trung bình của mỗi đoạn (Nhóm chúng em sử dụng phương pháp "correlation", hay là phương pháp "tương quan")
        pitches, magnitudes = librosa.piptrack(y=segment, sr = 44100)
        index = magnitudes[:, 0].argmax()
        pitch = pitches[index, 0] if magnitudes[index, 0] > 0 else 0
        result.append(pitch if pitch > 0 else 120)

    result = np.array(result)
    result = (result - np.mean(result)) / (np.std(result) + 1e-6)
    return result.tolist()