import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm


def compute_metrics(
    meta_path: Path, data_dir: Path, conv_window_size: int = 3
) -> list[list]:
    """
    Compute metrics of the data item and write them to the metadata yaml file.

    Return: list of metrics: list[score_id, ...]
    """
    meta = OmegaConf.load(meta_path)
    score_id = meta.score.id
    audio_path = data_dir / "audio" / f"{score_id}.wav"
    if not audio_path.exists():
        return
    duration = meta.youtube.duration

    # load data
    warp_path = np.load(data_dir / "warp_path" / f"{score_id}.npy")
    beat_times = np.load(data_dir / "beat_times_aligned" / f"{score_id}.npy")
    SM = np.load(data_dir / "similarity" / f"{score_id}.npz")
    chroma_trace = SM.get("chroma_cqt")
    tempogram_trace = SM.get("tempogram")
    midi_numpy = np.load(data_dir / "midi_numpy" / f"{score_id}.npy")
    # Compute metrics
    wp_std = np.std(warp_path[0] - warp_path[1])
    norm_wp_std = wp_std / duration
    beat_times = np.append(beat_times, duration)
    beat_times = beat_times[np.diff(beat_times, prepend=-1) > 0.1]
    beat_times_fluctuation = np.diff(np.diff(beat_times))
    chroma_min = chroma_trace.min()
    tempogram_min = tempogram_trace.min()
    note_density = len(midi_numpy) / duration
    norm_time_diff = abs(duration - meta.score.duration) / duration

    # write to yaml
    metrics = meta.metrics
    metrics.beat_times_fluctuation_median = float(
        np.median(np.abs(beat_times_fluctuation))
    )
    metrics.norm_wp_std = float(norm_wp_std)
    metrics.chroma_min_similarity = float(chroma_min)
    metrics.tempogram_min_similarity = float(tempogram_min)
    metrics.note_density = note_density
    metrics.norm_time_diff = norm_time_diff
    OmegaConf.save(meta, meta_path)

    return [
        score_id,
        metrics.opt_chroma_shift,
        metrics.beat_times_fluctuation_median,
        metrics.norm_wp_std,
        chroma_min,
        tempogram_min,
        note_density,
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
            "score_id",
            "opt_chroma_shift",
            "beat_times_fluctuation_median",
            "norm_wp_std",
            "chroma_min_similarity",
            "tempogram_min_similarity",
            "note_density",
            "norm_time_diff",
        ],
    )
    df.to_csv("metrics.csv", index=False)
