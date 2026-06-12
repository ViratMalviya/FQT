# =============================================================================
# src/preprocessing.py
# Audio preprocessing and MFCC feature extraction for the FQT framework.
# =============================================================================
# This module handles:
#   - Loading raw audio files from disk using librosa
#   - Extracting Mel-Frequency Cepstral Coefficients (MFCCs)
#   - Building a PyTorch-compatible dataset and DataLoader
#   - Visualizing MFCC heatmaps for inspection
# =============================================================================

import os
import numpy as np
import librosa
import librosa.display
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the global YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# MFCC EXTRACTION
# =============================================================================

def extract_mfcc(
    audio_path: str,
    n_mfcc: int = 20,
    max_len: int = 100,
    sample_rate: int = 22050
) -> np.ndarray:
    """
    Load an audio file and extract its MFCC feature matrix.

    The MFCC matrix is padded or truncated along the time axis to a fixed
    length so that all samples have a uniform shape for batch processing.
    Each coefficient vector is also z-score normalized to zero mean and
    unit variance, which stabilizes quantum angle embeddings.

    Args:
        audio_path (str): Absolute or relative path to a .wav / .mp3 file.
        n_mfcc (int):     Number of MFCC coefficients to extract (default: 20).
        max_len (int):    Target time-frame length after padding/truncation.
        sample_rate (int): Sample rate for librosa.load() (default: 22050 Hz).

    Returns:
        np.ndarray: Float32 MFCC array of shape (n_mfcc, max_len).
    """
    # Load audio as a mono waveform at the specified sample rate
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    # Compute MFCCs — shape: (n_mfcc, time_frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate the time axis to max_len
    if mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]
    else:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")

    # Z-score normalize each coefficient row independently
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

    return mfcc.astype(np.float32)


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class AudioDeepfakeDataset(Dataset):
    """
    PyTorch Dataset for binary audio deepfake classification.

    Expects a root directory with two subdirectories:
        <dataset_path>/real/   — contains genuine (bonafide) audio files
        <dataset_path>/fake/   — contains spoofed (deepfake) audio files

    Each audio file is converted to a fixed-shape MFCC tensor on-the-fly.

    Args:
        dataset_path (str): Path to the dataset root directory.
        n_mfcc (int):       Number of MFCC coefficients.
        max_len (int):      Target time-frame length.
        sample_rate (int):  Audio sample rate for loading.
    """

    SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".ogg")

    def __init__(
        self,
        dataset_path: str,
        n_mfcc: int = 20,
        max_len: int = 100,
        sample_rate: int = 22050
    ):
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.sample_rate = sample_rate
        self.samples = []  # List of (file_path, label) tuples

        # Collect real audio files → label 0
        real_dir = os.path.join(dataset_path, "real")
        if os.path.isdir(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith(self.SUPPORTED_EXTS):
                    self.samples.append((os.path.join(real_dir, fname), 0))

        # Collect fake audio files → label 1
        fake_dir = os.path.join(dataset_path, "fake")
        if os.path.isdir(fake_dir):
            for fname in os.listdir(fake_dir):
                if fname.lower().endswith(self.SUPPORTED_EXTS):
                    self.samples.append((os.path.join(fake_dir, fname), 1))

        if not self.samples:
            raise RuntimeError(
                f"No audio files found in {dataset_path}. "
                "Ensure the dataset has 'real/' and 'fake/' subdirectories."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        audio_path, label = self.samples[idx]
        mfcc = extract_mfcc(
            audio_path,
            n_mfcc=self.n_mfcc,
            max_len=self.max_len,
            sample_rate=self.sample_rate
        )
        # Add channel dimension: (1, n_mfcc, max_len) for CNN input
        mfcc_tensor = torch.tensor(mfcc).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return mfcc_tensor, label_tensor


def load_local_dataset(
    dataset_path: str,
    batch_size: int = 32,
    n_mfcc: int = 20,
    max_len: int = 100,
    sample_rate: int = 22050,
    shuffle: bool = True
) -> DataLoader:
    """
    Build a PyTorch DataLoader from a local audio dataset directory.

    Args:
        dataset_path (str): Path to dataset root (must contain real/ and fake/).
        batch_size (int):   Number of samples per mini-batch.
        n_mfcc (int):       Number of MFCC coefficients.
        max_len (int):      MFCC time-frame length.
        sample_rate (int):  Audio sample rate.
        shuffle (bool):     Whether to shuffle data each epoch.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = AudioDeepfakeDataset(
        dataset_path=dataset_path,
        n_mfcc=n_mfcc,
        max_len=max_len,
        sample_rate=sample_rate
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_mfcc_heatmap(
    audio_path: str,
    n_mfcc: int = 20,
    sample_rate: int = 22050,
    save_path: str = None
):
    """
    Generate a 4-panel MFCC inspection heatmap for a given audio file.

    Panels:
        1. Raw waveform (amplitude vs. time)
        2. Log-mel spectrogram
        3. MFCC coefficients (n_mfcc × time)
        4. MFCC delta (first derivative — captures temporal dynamics)

    Args:
        audio_path (str): Path to the audio file.
        n_mfcc (int):     Number of MFCC coefficients.
        sample_rate (int): Sample rate for loading.
        save_path (str):  If provided, saves the figure to this path (PNG).
    """
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("MFCC & Audio Feature Inspection", fontsize=14, fontweight="bold")

    # Panel 1: Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0, 0])
    axes[0, 0].set_title("Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    # Panel 2: Log-mel spectrogram
    img2 = librosa.display.specshow(
        mel_spec_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[0, 1]
    )
    axes[0, 1].set_title("Log-Mel Spectrogram")
    fig.colorbar(img2, ax=axes[0, 1], format="%+2.0f dB")

    # Panel 3: MFCC heatmap
    img3 = librosa.display.specshow(
        mfcc, sr=sr, x_axis="time", ax=axes[1, 0]
    )
    axes[1, 0].set_title(f"MFCC ({n_mfcc} coefficients)")
    axes[1, 0].set_ylabel("MFCC Coefficient")
    fig.colorbar(img3, ax=axes[1, 0])

    # Panel 4: MFCC delta (temporal derivative)
    img4 = librosa.display.specshow(
        mfcc_delta, sr=sr, x_axis="time", ax=axes[1, 1]
    )
    axes[1, 1].set_title("MFCC Delta (Δ)")
    axes[1, 1].set_ylabel("MFCC Coefficient")
    fig.colorbar(img4, ax=axes[1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] MFCC heatmap saved to: {save_path}")
    else:
        plt.show()

    plt.close()
