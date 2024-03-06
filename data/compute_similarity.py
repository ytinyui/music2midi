import argparse
import multiprocessing
import os
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from pretty_midi import PrettyMIDI
from tqdm import tqdm

from src.dsp import pad_audio

"""
*Run this script as a module.
*Example: python -m data.compute_similarity.py path/to/dataset/directory
"""


def chunk_avg(x: np.ndarray, chunk_size: int) -> np.ndarray:
    splits = np.array_split(x, x.shape[-1] // chunk_size, axis=1)
    return np.stack([np.mean(col, axis=1) for col in splits], axis=1)


def cosine_similarity(
    x: np.ndarray,
    y: np.ndarray,
    diag_only: bool = False,
) -> np.ndarray:
    """
    Compute the similarity matrix.

    If diag_only == True, only the diagonal of the similarity matrix is computed
    """
    x_norm = np.linalg.norm(x, axis=0)
    y_norm = np.linalg.norm(y, axis=0)
    x = x / (x_norm[None, :] + 1e-8)
    y = y / (y_norm[None, :] + 1e-8)

    if diag_only:
        return np.sum(x * y, axis=0)
    else:
        return np.dot(x.T, y)


def compute_sm(
    x: np.ndarray,
    y: np.ndarray,
    feature: Callable,
    hop_length: int,
    chunk_size: int,
) -> np.ndarray:
    x_feat = feature(y=x, sr=sr, hop_length=hop_length)
    y_feat = feature(y=y, sr=sr, hop_length=hop_length)
    x_feat = chunk_avg(x_feat, chunk_size)
    y_feat = chunk_avg(y_feat, chunk_size)
    return cosine_similarity(x_feat, y_feat, diag_only=True)


def main(
    song_path: Path,
    data_dir: Path,
    sr: int = 22050,
    hop_length: int = 490,
    chunk_size: int = 45,
):
    score_id = song_path.stem
    midi_path = os.path.join(data_dir, "midi_aligned", score_id + ".mid")
    if not song_path.exists():
        print(f"{song_path.name} file not found")
        return
    if not os.path.exists(midi_path):
        print(f"{os.path.basename(midi_path)} file not found")
        return

    song_audio, sr = librosa.load(str(song_path), sr=sr)
    midi_data = PrettyMIDI(midi_path)
    midi_synth = midi_data.fluidsynth(fs=sr)
    song_audio, midi_synth = pad_audio(song_audio, midi_synth)

    SM_chroma_cqt = compute_sm(
        song_audio,
        midi_synth,
        librosa.feature.chroma_cqt,
        hop_length=hop_length,
        chunk_size=chunk_size,
    )
    SM_tempogram = compute_sm(
        song_audio,
        midi_synth,
        librosa.feature.tempogram,
        hop_length=hop_length,
        chunk_size=chunk_size,
    )

    np.savez(
        os.path.join(data_dir, "similarity", score_id + ".npz"),
        chroma_cqt=SM_chroma_cqt,
        tempogram=SM_tempogram,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    os.makedirs(data_dir / "similarity", exist_ok=True)
    config = OmegaConf.load("config.yaml")
    sr = config.dataset.sample_rate

    Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
        delayed(main)(song_path, data_dir, sr=sr)
        for song_path in tqdm(list(data_dir.glob("audio/*.wav")))
    )
