import argparse
import multiprocessing
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from pretty_midi import PrettyMIDI
from tqdm import tqdm


def pad_audio(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the shorter audio to the longer one.
    Returns a tuple of the two arrays.
    """
    return (
        np.pad(x, (0, max(len(y) - len(x), 0))),
        np.pad(y, (0, max(len(x) - len(y), 0))),
    )


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
    sr: int,
    hop_length: int = 512,
    chunk_size: int = 96,
):
    score_id = song_path.stem
    midi_path = data_dir / "midi_aligned" / f"{score_id}.mid"
    output_path = data_dir / "similarity" / f"{score_id}.npz"
    if output_path.exists():
        print(f"{output_path} already exists")
        return
    if not song_path.exists():
        print(f"{song_path} file not found")
        return
    if not midi_path.exists():
        print(f"{midi_path} file not found")
        return

    song_audio, sr = librosa.load(str(song_path), sr=sr)
    midi_data = PrettyMIDI(str(midi_path))
    midi_synth = midi_data.synthesize(fs=sr)
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
        output_path,
        chroma_cqt=SM_chroma_cqt,
        tempogram=SM_tempogram,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    (data_dir / "similarity").mkdir(exist_ok=True)
    config = OmegaConf.load(args.config)
    sr = config.dataset.sample_rate
    chunk_size = config.dataset.similarity_chunk_size

    Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
        delayed(main)(song_path, data_dir, sr=sr, chunk_size=chunk_size)
        for song_path in tqdm(list(data_dir.glob("audio_preprocessed/*.wav")))
    )
