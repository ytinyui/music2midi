import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from pretty_midi import PrettyMIDI
from tqdm import tqdm


def rms(arr: np.ndarray):
    return np.sqrt(np.mean(arr**2))


def compute_metrics(meta_path: Path, data_dir: Path) -> list:
    """
    Compute metrics of the data item and write them to the metadata yaml file.

    Return: list of metrics: list[piano_id, ...]
    """
    meta = OmegaConf.load(meta_path)
    piano_id = meta.piano.id
    audio_path = data_dir / "audio" / f"{piano_id}.wav"
    if not audio_path.exists():
        return
    duration = meta.youtube.duration

    # load data
    warp_path = np.load(data_dir / "warp_path" / f"{piano_id}.npy")
    beat_times = np.load(data_dir / "beat_times_aligned" / f"{piano_id}.npy")
    midi_data = PrettyMIDI(str(data_dir / "midi_transposed" / f"{piano_id}.mid"))
    numpy_notes = np.load(data_dir / "midi_numpy" / f"{piano_id}.npy")
    # Compute metrics
    wp_std = np.std(warp_path[0] - warp_path[1])

    beat_times = np.append(beat_times, duration)
    # remove beats shorter than 0.1s
    beat_times = beat_times[np.diff(beat_times, prepend=-1) > 0.1]
    split_count = 10
    beat_times_split = np.array_split(beat_times, split_count)
    max_beat_fluctuation = np.max(
        [rms(np.diff(np.diff(x))) for x in beat_times_split if len(x) > 2]
    )

    notes_split_indices = np.searchsorted(
        numpy_notes[:, 0], [x[0] for x in beat_times_split if len(x) > 1]
    )
    notes_split = np.array_split(numpy_notes, notes_split_indices)
    duration_split = [x[-1] - x[0] for x in beat_times_split if len(x) > 1]
    max_note_density = np.max(
        [len(notes) / dur for (notes, dur) in zip(notes_split, duration_split)]
    )

    midi_duration = midi_data.get_end_time()
    time_diff_ratio = abs(duration - midi_duration) / duration

    # write to yaml
    metrics = meta.metrics
    metrics.wp_std = float(wp_std)
    metrics.max_beat_fluctuation = float(max_beat_fluctuation)
    metrics.max_note_density = float(max_note_density)
    metrics.time_diff_ratio = float(time_diff_ratio)
    OmegaConf.save(meta, meta_path)

    return [
        str(piano_id),
        metrics.opt_chroma_shift,
        wp_std,
        max_beat_fluctuation,
        max_note_density,
        time_diff_ratio,
        meta.piano.genre,
        meta.piano.difficulty,
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config = OmegaConf.load(args.config)
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
            "wp_std",
            "max_beat_fluctuation",
            "max_note_density",
            "time_diff_ratio",
            "genre",
            "difficulty",
        ],
    )
    df.to_csv("metrics.csv", index=False)
