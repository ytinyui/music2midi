from typing import Iterable

import librosa
import mir_eval
import numba
import numpy as np
from pretty_midi import PrettyMIDI


@numba.njit()
def get_highest_pitches_from_piano_roll(piano_roll: np.ndarray) -> np.ndarray:
    num_frames = piano_roll.shape[1]
    ret = np.empty(num_frames, dtype=np.int_)
    for i in range(num_frames):
        if np.sum(piano_roll[:, i]) == 0:  # no onset at this frame
            ret[i] = np.nan
        (onset_pitches,) = np.nonzero(piano_roll[:, i])
        ret[i] = onset_pitches[-1]

    return ret


def extract_midi_melody(
    target: PrettyMIDI, output: PrettyMIDI, fs: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    return two arrays, each column of the array is the pitch class of the highest pitch onset at each frame.
    """
    target_end_time = target.get_end_time()
    output_end_time = output.get_end_time()
    end_time = max(output_end_time, target_end_time)
    times = np.arange(0, end_time, 1 / fs)

    target_piano_roll = target.get_piano_roll(fs=fs, times=times)
    output_piano_roll = output.get_piano_roll(fs=fs, times=times)
    target = get_highest_pitches_from_piano_roll(target_piano_roll)
    output = get_highest_pitches_from_piano_roll(output_piano_roll)

    if len(target) == 0 and len(output) > 0:
        target = np.zeros_like(output, dtype=np.int_)
    if len(output) == 0 and len(target) > 0:
        output = np.zeros_like(target, dtype=np.int_)

    return target, output


def melody_chroma_accuracy(
    ref_pitch: np.ndarray, est_pitch: np.ndarray, fs: int = 100
) -> np.ndarray:
    assert ref_pitch.shape[0] == len(ref_pitch)
    assert ref_pitch.shape == est_pitch.shape

    ref_freq = librosa.midi_to_hz(ref_pitch)
    est_freq = librosa.midi_to_hz(est_pitch)
    times = np.arange(0, len(ref_pitch)) * 1 / fs

    ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(
        times, ref_freq, times, est_freq
    )

    return mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)


def evaluate_batch(
    targets: Iterable[PrettyMIDI],
    outputs: Iterable[PrettyMIDI],
) -> float:
    data = [
        extract_midi_melody(target, output) for target, output in zip(targets, outputs)
    ]
    targets, outputs = zip(*data)
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)

    return melody_chroma_accuracy(targets, outputs)
