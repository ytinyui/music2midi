import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm


def compute_metrics(meta_path: Path, data_dir: Path) -> list[list]:
    """
    Compute metrics of the data item and write them to the metadata yaml file.

    Return: list of metrics: list[piano_id, ...]
    """
    meta = OmegaConf.load(meta_path)
    piano_id = meta.piano.id
    audio_path = data_dir / "audio_preprocessed" / f"{piano_id}.wav"
    if not audio_path.exists():
        return
    duration = meta.youtube.duration

    # load data
    warp_path = np.load(data_dir / "warp_path" / f"{piano_id}.npy")
    beat_times = np.load(data_dir / "beat_times_aligned" / f"{piano_id}.npy")
    SM = np.load(data_dir / "similarity" / f"{piano_id}.npz")
    chroma_similarity = SM.get("chroma_cqt")
    tempogram_similarity = SM.get("tempogram")
    numpy_notes = np.load(data_dir / "midi_numpy" / f"{piano_id}.npy")
    # Compute metrics
    wp_std = np.std(warp_path[0] - warp_path[1])
    norm_wp_std = wp_std / duration

    beat_times = np.append(beat_times, duration)
    beat_times = beat_times[np.diff(beat_times, prepend=-1) > 0.1]
    split_count = 10
    beat_times_split = np.array_split(beat_times, split_count)
    beat_local_fluctuation = np.mean(
        [np.max(np.abs(np.diff(np.diff(x)))) for x in beat_times_split if len(x) > 2]
    )
    notes_split_indices = np.searchsorted(
        numpy_notes[:, 0], [x[0] for x in beat_times_split if len(x) > 0]
    )
    notes_split = np.array_split(numpy_notes, notes_split_indices)
    piecewise_note_density = (
        np.mean([len(x) for x in notes_split]) * split_count / duration
    )

    chroma_min = chroma_similarity.min()
    tempogram_min = tempogram_similarity.min()
    norm_time_diff = abs(duration - meta.piano.duration) / duration

    # write to yaml
    metrics = meta.metrics
    metrics.norm_wp_std = float(norm_wp_std)
    metrics.beat_local_fluctuation = float(beat_local_fluctuation)
    metrics.piecewise_note_density = float(piecewise_note_density)
    metrics.chroma_min_similarity = float(chroma_min)
    metrics.tempogram_min_similarity = float(tempogram_min)
    metrics.norm_time_diff = norm_time_diff
    OmegaConf.save(meta, meta_path)

    return [
        piano_id,
        metrics.opt_chroma_shift,
        norm_wp_std,
        beat_local_fluctuation,
        piecewise_note_density,
        chroma_min,
        tempogram_min,
        norm_time_diff,
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    metrics_list = Parallel(
        n_jobs=multiprocessing.cpu_count() // 2, backend="multiprocessing"
    )(
        delayed(compute_metrics)(meta_path, data_dir)
        for meta_path in tqdm(list(data_dir.glob("metadata/*.yaml")))
    )

    metrics_list = [x for x in metrics_list if x is not None]
    df = pd.DataFrame(
        metrics_list,
        columns=[
            "piano_id",
            "opt_chroma_shift",
            "norm_wp_std",
            "beat_local_fluctuation",
            "piecewise_note_density",
            "chroma_min_similarity",
            "tempogram_min_similarity",
            "norm_time_diff",
        ],
    )
    df.to_csv("metrics.csv", index=False)
