import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample

target_sample_rate = 44100


def extract_features(audio_file, n_fft=2048, hop_length=512, n_mfcc=20):
    try:
        # Load audio file
        y, sr = sf.read(audio_file)

        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Resample to target sample rate
        if sr != target_sample_rate:
            y = resample(y, int(len(y) * target_sample_rate / sr))
            sr = target_sample_rate

        # Take first 5 seconds
        samples_5_sec = target_sample_rate * 5
        y_5_sec = y[:samples_5_sec]
        if len(y_5_sec) < samples_5_sec:
            y_5_sec = np.pad(y_5_sec, (0, samples_5_sec - len(y_5_sec)))

        # Extract MFCC
        mfccs = librosa.feature.mfcc(
            y=y_5_sec, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # Extract delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        # Combine features
        combined = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)

        # Normalize and compute statistics
        mean_features = np.mean(combined, axis=1)
        std_features = np.std(combined, axis=1)

        return np.concatenate([mean_features, std_features])

    except Exception as e:
        print(f"Error processing file {audio_file}: {e}")
        return None
