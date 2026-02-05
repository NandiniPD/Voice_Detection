import os
import glob
import io
import json
from pathlib import Path

import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_audio_array(path, sr=16000):
    # Use pydub to open many formats then resample
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(sr).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if samples.size == 0:
        return None, None
    samples /= np.max(np.abs(samples))
    return samples, sr


def featurize(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    flatness = librosa.feature.spectral_flatness(y=y)
    rms = librosa.feature.rms(y=y)
    delta_mfcc = librosa.feature.delta(mfcc)
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_std = float(np.nanstd(f0))
        voiced_fraction = float(np.mean(~np.isnan(f0)))
    except Exception:
        pitch_std = float('nan')
        voiced_fraction = 0.0

    feats = {
        "mfcc_std": float(np.std(mfcc)),
        "zcr_mean": float(np.mean(zcr)),
        "flatness_mean": float(np.mean(flatness)),
        "pitch_std": pitch_std,
        "voiced_fraction": voiced_fraction,
        "rms_std": float(np.std(rms)),
        "delta_mfcc_std": float(np.std(delta_mfcc)),
    }
    return feats


def collect_dataset(ai_dir="data/ai", human_dir="data/human"):
    X = []
    y = []
    feature_order = None

    for label, d in [(1, ai_dir), (0, human_dir)]:
        for filepath in glob.glob(os.path.join(d, "**/*.*"), recursive=True):
            try:
                arr, sr = load_audio_array(filepath)
                if arr is None:
                    print("Skipping empty", filepath)
                    continue
                feats = featurize(arr, sr)
                if feature_order is None:
                    feature_order = list(feats.keys())
                X.append([feats[k] for k in feature_order])
                y.append(label)
                print("Collected", filepath)
            except Exception as e:
                print("Failed to process", filepath, e)

    return np.array(X, dtype=float), np.array(y, dtype=int), feature_order


def train(ai_dir="data/ai", human_dir="data/human", out_model="models/model.joblib"):
    X, y, feature_order = collect_dataset(ai_dir, human_dir)
    if X.size == 0:
        print("No data found. Put labeled audio files under data/ai and data/human")
        return

    # simple fill for NaNs
    inds = np.isnan(X)
    if inds.any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(clf, out_model)
    with open(str(MODEL_DIR / "feature_order.json"), "w") as f:
        json.dump(feature_order, f)
    print("Saved model to", out_model)


if __name__ == "__main__":
    train()
