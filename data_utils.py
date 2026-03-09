import os
import librosa
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def extract_mfcc(path, n_mfcc=20, max_len=100):
    y, sr = librosa.load(path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.astype(np.float32)

def load_local_dataset(base_path, batch_size=16):
    X, y = [], []
    for label, cls in enumerate(["REAL", "FAKE"]):
        folder = os.path.join(base_path, cls)
        if not os.path.exists(folder):
            print(f"Warning: folder '{folder}' not found, skipping.")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith((".mp3", ".wav"))]
        print(f"Found {len(files)} files in '{folder}'")

        for f in files:
            path = os.path.join(folder, f)
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                print(f"Skipping missing/empty: {path}")
                continue
            try:
                mfcc = extract_mfcc(path)
                X.append(mfcc)
                y.append(label)
            except Exception as e:
                print(f"Skipping {f}: {e}")
                continue

    if len(X) == 0:
        raise RuntimeError("No valid audio files found.")

    dataset = TensorDataset(torch.tensor(np.array(X)), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)