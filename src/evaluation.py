from functools import partial
from typing import Callable, Iterable

import numba
import numpy as np
import sklearn.metrics as metrics
from pretty_midi import PrettyMIDI


@numba.njit()
def get_highest_pitch_from_each_frame(piano_roll: np.ndarray) -> np.ndarray:
    """
    return the pitch class (0, 1, ..., 11) of the highest pitch onset at each frame.
    if there is no onset at this frame, return 12.

    input:
        piano_roll : (128, num_frames)

    return: (num_frames, )
    """
    num_frames = piano_roll.shape[1]
    ret = np.ones(num_frames, dtype=np.int_) * 12
    for i in range(num_frames):
        if np.sum(piano_roll[:, i]) == 0:  # no onset at this frame
            continue
        (onset_pitches,) = np.nonzero(piano_roll[:, i])
        ret[i] = onset_pitches[-1] % 12

    return ret


def midi_to_classes_arr(
    target: PrettyMIDI, output: PrettyMIDI, piano_roll_fs: int
) -> np.ndarray:
    """
    return an array, where each column is the pitch class of the highest pitch onset at each frame.
    """
    target_end_time = target.get_end_time()
    output_end_time = output.get_end_time()
    end_time = max(output_end_time, target_end_time)
    times = np.arange(0, end_time, 1 / piano_roll_fs)

    target_piano_roll = target.get_piano_roll(fs=piano_roll_fs, times=times)
    output_piano_roll = output.get_piano_roll(fs=piano_roll_fs, times=times)
    target = get_highest_pitch_from_each_frame(target_piano_roll)
    output = get_highest_pitch_from_each_frame(output_piano_roll)

    if len(target) == 0 and len(output) > 0:
        target = np.zeros_like(output, dtype=np.int_)
    if len(output) == 0 and len(target) > 0:
        output = np.zeros_like(target, dtype=np.int_)

    return target, output


def metric_fn_wrapper(
    y_true: np.ndarray, y_pred: np.ndarray, fn: Callable, **kwargs
) -> float:
    if len(y_true) == len(y_pred) == 0:
        return 1.0
    return fn(y_true, y_pred, **kwargs)


def evaluate_batch(
    targets: Iterable[PrettyMIDI],
    outputs: Iterable[PrettyMIDI],
    piano_roll_fs: int = 100,
) -> dict[str, float]:
    data = [
        midi_to_classes_arr(target, output, piano_roll_fs)
        for target, output in zip(targets, outputs)
    ]
    targets, outputs = zip(*data)
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)

    metrics_fn = {
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
        "f1": metrics.f1_score,
    }
    metrics_fn = {
        k: partial(metric_fn_wrapper, fn=fn, average="micro")
        for k, fn in metrics_fn.items()
    }

    return {k: fn(targets, outputs) for k, fn in metrics_fn.items()}
