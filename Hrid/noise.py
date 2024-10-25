import os
import librosa
import noisereduce as nr
import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
from pydub import AudioSegment

# Helper functions for audio enhancement
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def dynamic_range_compression(audio, threshold=-30.0, ratio=4.0):
    audio_db = librosa.amplitude_to_db(audio)
    compressed_audio = np.where(audio_db > threshold, (1 / ratio) * (audio_db - threshold) + threshold, audio_db)
    return librosa.db_to_amplitude(compressed_audio)

# Process multiple files for advanced enhancement
def process_audio_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load the audio file
            y, sr = librosa.load(input_path, sr=None)

            # Apply noise reduction using a deep learning model or NR library
            reduced_noise = nr.reduce_noise(y=y, sr=sr)

            # Apply bandpass filter to keep speech frequencies (300Hz - 3400Hz)
            filtered_audio = bandpass_filter(reduced_noise, lowcut=300.0, highcut=3400.0, fs=sr)

            # Apply dynamic range compression to enhance clarity
            compressed_audio = dynamic_range_compression(filtered_audio)

            # Normalize the audio
            normalized_audio = librosa.util.normalize(compressed_audio)

            # Save the enhanced audio
            write(output_path, sr, (normalized_audio * 32767).astype(np.int16))

            print(f"Processed: {filename}")

# Main function
input_directory = "input_audio_files"
output_directory = "enhanced_audio_files"
process_audio_files(input_directory, output_directory)
