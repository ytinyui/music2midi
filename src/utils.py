from typing import Literal

import numpy as np
import pretty_midi


def pad_audio(
    x: np.ndarray, y: np.ndarray, mode: Literal["max_length", "fix_x"] = "max_length"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the shorter audio to the longer one, or truncate/pad y to match the length of x
    Returns a tuple of the two arrays.
    """
    if mode == "max_length":
        return (
            np.pad(x, (0, max(len(y) - len(x), 0))),
            np.pad(y, (0, max(len(x) - len(y), 0))),
        )
    if mode == "fix_x":
        if len(y) > len(x):
            return x, y[: len(x)]
        return x, np.pad(y, (0, len(x) - len(y)))
    raise ValueError


def to_stereo(x, y):
    x, y = pad_audio(x, y)
    return np.stack((x, y))


def numpy_to_midi(notes: np.ndarray) -> pretty_midi.PrettyMIDI:
    midi_data = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
    new_inst = pretty_midi.Instrument(program=0, name="Piano")

    new_inst.notes = [
        pretty_midi.Note(
            start=onset_time,
            end=offset_time,
            pitch=int(pitch),
            velocity=int(velocity),
        )
        for onset_time, offset_time, pitch, velocity in notes
    ]
    midi_data.instruments.append(new_inst)
    midi_data.remove_invalid_notes()
    return midi_data
