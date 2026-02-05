#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
import librosa
from pydub import AudioSegment
import numpy as np
from pathlib import Path

def load_audio(path):
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if np.max(np.abs(samples)) > 0:
        samples /= np.max(np.abs(samples))
    return samples, 16000

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    flatness = librosa.feature.spectral_flatness(y=y)
    rms = librosa.feature.rms(y=y)
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_std = float(np.nanstd(f0))
        voiced_fraction = float(np.mean(~np.isnan(f0)))
    except:
        pitch_std = float(np.nan)
        voiced_fraction = float(np.mean(~np.isnan(librosa.core.pitch.piptrack(y=y, sr=sr)[0])))
    
    return {
        "mfcc_std": float(np.std(mfcc)),
        "zcr_mean": float(np.mean(zcr)),
        "flatness_mean": float(np.mean(flatness)),
        "pitch_std": float(pitch_std),
        "voiced_fraction": float(voiced_fraction),
        "rms_std": float(np.std(rms)),
        "delta_mfcc_std": float(np.std(librosa.feature.delta(mfcc))),
    }

for file in ["data/human/11.wav", "data/human/Hasan_hum.wav"]:
    y, sr = load_audio(file)
    features = extract_features(y, sr)
    print(f"{Path(file).name}:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")
    print()
