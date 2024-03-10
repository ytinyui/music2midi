import argparse
import multiprocessing
from pathlib import Path

import librosa
import numpy as np
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm


def interp_beat_times(beat_times: np.ndarray, sub_beats: int) -> np.ndarray:
    """
    Divide each beat into sub-beats
    Return: an array of time of each sub-beat
    """
    if sub_beats == 1:
        return beat_times
    return np.interp(
        np.arange(0, beat_times.size * sub_beats)[: 1 - sub_beats],
        np.arange(0, beat_times.size) * sub_beats,
        beat_times,
    )


def quantize_note_times(input: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    """
    Quantize note times to the nearest beat.
    """
    bins = (beat_times[1:] + beat_times[:-1]) / 2
    return beat_times[np.digitize(input, bins)]


def main(meta_path: Path, data_dir: Path, sub_beats: int, sr: int):
    assert sub_beats > 0
    meta = OmegaConf.load(meta_path)
    score_id = meta.score.id
    beat_times_path = data_dir / "beat_times_aligned" / f"{score_id}.npy"
    if not beat_times_path.exists():
        # print(f"{beat_times_path} file not found")
        return

    beat_times = np.load(beat_times_path)
    numpy_notes = np.load(data_dir / "midi_numpy" / f"{score_id}.npy")
    audio_duration = librosa.get_duration(
        path=str(data_dir / "audio_preprocessed" / f"{score_id}.wav"), sr=sr
    )
    beat_times = np.append(beat_times, audio_duration)
    # beat time interval lower limit: 100ms
    beat_times = beat_times[np.diff(beat_times, prepend=-1) > 0.1]
    beat_times_interpolated = interp_beat_times(beat_times, sub_beats)

    try:
        onset_times = quantize_note_times(numpy_notes[:, 0], beat_times_interpolated)
        offset_times = quantize_note_times(numpy_notes[:, 1], beat_times_interpolated)
    except ValueError as e:
        print(f"{e} in {score_id}")
        raise
    onset_time_indices = np.searchsorted(beat_times_interpolated, onset_times)
    offset_time_indices = np.searchsorted(beat_times_interpolated, offset_times)

    numpy_notes_quantized = np.stack(
        [onset_time_indices, offset_time_indices, numpy_notes[:, 2], numpy_notes[:, 3]],
    ).T
    numpy_notes_quantized = np.int_(numpy_notes_quantized)
    # min length of each note is 1 step
    numpy_notes_quantized = numpy_notes_quantized[
        numpy_notes_quantized[:, 1] - numpy_notes_quantized[:, 0] > 0
    ]
    np.save(
        data_dir / "midi_quantized_numpy" / f"{score_id}.npy", numpy_notes_quantized
    )
    np.save(
        data_dir / "beat_times_interpolated" / f"{score_id}.npy",
        beat_times_interpolated,
    )
    OmegaConf.save(meta, meta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    (data_dir / "midi_quantized_numpy").mkdir(exist_ok=True)
    (data_dir / "beat_times_interpolated").mkdir(exist_ok=True)
    config = OmegaConf.load(args.config)
    sub_beats = config.dataset.quantize_sub_beats
    sr = config.dataset.sample_rate

    Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
        delayed(main)(meta_path, data_dir, sub_beats, sr)
        for meta_path in tqdm(list(data_dir.glob("metadata/*.yaml")))
    )
