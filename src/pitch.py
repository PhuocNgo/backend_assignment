import numpy as np
import librosa


def funcPitch(audio_path):
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=44100, mono=True)

        # Take first 5 seconds
        samples_5_sec = sr * 5
        y = y[:samples_5_sec]
        if len(y) < samples_5_sec:
            y = np.pad(y, (0, samples_5_sec - len(y)))

        # Extract pitch using pyin
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=512)

        # Filter valid pitch values
        valid_f0 = f0[voiced_flag]
        if len(valid_f0) == 0:
            return np.zeros(4)

        # Compute statistics
        mean_pitch = np.mean(valid_f0)
        std_pitch = np.std(valid_f0)
        min_pitch = np.min(valid_f0)
        max_pitch = np.max(valid_f0)

        return np.array([mean_pitch, std_pitch, min_pitch, max_pitch])

    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return np.zeros(4)
