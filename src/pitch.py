from pydub import AudioSegment
import numpy as np


def funcPitch_pydub(path):
    try:
        audio = AudioSegment.from_wav(path)
    except Exception as e:
        print(f"Lỗi khi mở file {path}: {e}")
        return [0] * 7

    # Chuyển đổi thành mono và lấy tín hiệu âm thanh
    audio = audio.set_channels(1)  # Đảm bảo là mono
    audio = audio.set_frame_rate(44100)  # Đảm bảo tần số mẫu là 44100
    samples = np.array(audio.get_array_of_samples())

    # Chia tín hiệu thành các đoạn nhỏ
    num_segments = 7
    segment_length = len(samples) // num_segments
    result = []

    for i in range(num_segments):
        segment = samples[i * segment_length: (i + 1) * segment_length]
        # Trích xuất tần số trung bình của mỗi đoạn (có thể sử dụng FFT hoặc các phương pháp khác)
        # Đơn giản cho ví dụ, có thể thay thế bằng phương pháp khác
        mean_pitch = float(np.mean(segment))
        result.append(mean_pitch)

    return result
