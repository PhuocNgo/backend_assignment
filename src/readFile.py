import os
import wave

def list_wav_files(path):
    wav_files = []
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            wav_files.append(filename)
    return wav_files
