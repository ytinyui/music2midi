from __future__ import annotations

from typing import Optional

import numpy as np
import pretty_midi
from numba import njit
from omegaconf import DictConfig

PAD = 0
EOS = 1


class MidiTokenizer:
    def __init__(self, config: DictConfig):
        self.config = config.tokenizer

        if config.dataset.quantize_sub_beats:
            self.time_step = 0
        else:
            self.time_step = self.config.midi_quantize_ms / 1000
        self.reserved_token_offset = self.config.vocab_size.special
        self.pitch_token_offset = (
            self.reserved_token_offset + self.config.vocab_size.reserved
        )
        self.velocity_token_offset = (
            self.pitch_token_offset + self.config.vocab_size.pitch
        )
        self.time_token_offset = (
            self.velocity_token_offset + self.config.vocab_size.velocity
        )

    def to_string(self, tokens: np.ndarray) -> list[str]:
        def _to_string(token: int) -> str:
            if token == PAD:
                return "PAD"
            if token == EOS:
                return "EOS"
            if token >= self.time_token_offset:
                return ("time", token - self.time_token_offset)
            if token >= self.velocity_token_offset:
                return ("velocity", token - self.velocity_token_offset)
            if token >= self.pitch_token_offset:
                return ("note", token - self.pitch_token_offset)
            if token >= self.reserved_token_offset:
                return ("reserved", token - self.reserved_token_offset)

            raise ValueError(f"Invalid token '{token}'")

        return [_to_string(token) for token in tokens]

    def batch_decode(
        self,
        batched_tokens: list[np.ndarray],
        duration_per_batch: float,
        cutoff_time: Optional[int] = None,
    ) -> np.ndarray:
        """
        Use this method only if second is time unit

        input: list of np.ndarray tokens
        order: batch then seq_len
            - example: input[0  , 0], input[0, 1], ..., input[0, L-1],
                       input[1  , 0], input[1, 1], ..., input[1, L-1],
                       ...
                       input[N-1, 0], input[N, 1], ..., input[N, L-1]

        duration_per_batch: duration of each batch (in seconds)
        return: [L']
        """

        ret = []
        start_idx = 0
        n_steps = round(duration_per_batch / self.time_step)
        for tokens in batched_tokens:
            notes = self.decode(tokens, start_idx=start_idx, cutoff_time=cutoff_time)
            ret.append(notes)
            start_idx += n_steps

        return np.concatenate(ret)

    def __call__(
        self,
        notes: np.ndarray,
        start_time: Optional[int] = None,
        cutoff_time: Optional[int] = None,
        add_eos: Optional[bool] = True,
    ) -> np.ndarray:
        """
        start_time: optional. Default: the time of the first onset
        cutoff_time: optional. Drop all notes after cutoff_time

        notes[0]: (onset_time, offset_time, pitch, velocity)

        Notes:
            - if start time is None, the time is aligned with the first onset
            - if add_eos is True, add EOS token at the end of the tokens list
            - if the relative time steps exceed vocab size, the time steps will be clipped.
        """
        if len(notes) == 0:
            tokens = np.array([])

        else:
            notes = np.copy(notes)
            if start_time is None:
                start_time = notes[0, 0]
            notes[:, :2] -= start_time

            if cutoff_time is not None:
                notes = notes[notes[:, 0] < cutoff_time]

            if self.time_step:
                # min length of each note is 1 step
                notes[:, 1] = np.maximum(notes[:, 1], notes[:, 0] + self.time_step)
                # convert time to indices
                notes[:, :2] = notes[:, :2] / self.time_step
                notes[:, :2] = np.rint(np.nextafter(notes[:, :2], notes[:, :2] + 1))
            # clip time step values to fit vocab size
            notes[:, :2] = np.minimum(notes[:, :2], self.config.vocab_size.time - 1)

            time_indices = np.unique(notes[:, :2])
            tokens = np.array(
                [
                    tok
                    for index in time_indices
                    for tok in self._get_tokens(
                        onset_notes=get_onset_notes(notes, index),
                        offset_notes=get_offset_notes(notes, index),
                        index=index,
                    )
                ]
            )

        if add_eos:
            tokens = np.append(tokens, EOS)

        return tokens.astype(np.int_)

    def decode(
        self,
        tokens: np.ndarray,
        start_idx: int = 0,
        cutoff_time: Optional[int] = None,
    ) -> np.ndarray:
        """
        Decode tokens into notes array.
        If cutoff_time is provided, the note events after cutoff_time are ignored.
        """
        notes = self._decode_tokens(tokens, start_idx)
        # remove notes without offset time
        notes = notes[notes[:, 1] != -1]
        # convert time index to seconds
        if self.time_step:
            notes[:, :2] = notes[:, :2] * self.time_step
        else:
            notes = np.int_(notes)

        if cutoff_time is not None:
            # drop note onsets beyond cutoff_time
            notes = notes[notes[:, 0] < cutoff_time]
            # truncate offsets
            notes[:, 1] = np.where(notes[:, 1] > cutoff_time, cutoff_time, notes[:, 1])

        return notes

    def _decode_tokens(self, tokens: np.ndarray, start_idx: int) -> np.ndarray:
        """
        Decode tokens to numpy array (pretty_midi format)
        """
        notes = np.zeros((0, 4))
        cur_time_idx = -1
        cur_velocity = -1
        cur_note = -1

        for token in tokens:
            if token == EOS:
                break
            if token >= self.time_token_offset:
                cur_time_idx = start_idx + token - self.time_token_offset
                cur_velocity = -1
                cur_note = -1
            elif token >= self.velocity_token_offset:
                cur_velocity = token - self.velocity_token_offset
                cur_velocity *= self.config.default_velocity
                cur_note = -1
            elif token >= self.pitch_token_offset:
                cur_note = token - self.pitch_token_offset
            elif token >= self.reserved_token_offset:
                continue

            if -1 in [cur_time_idx, cur_velocity, cur_note]:
                continue

            notes = _tokens_to_note(notes, cur_time_idx, cur_note, cur_velocity)
            cur_note = -1

        return notes.astype(np.float_)

    def _get_tokens(
        self, onset_notes: np.ndarray, offset_notes: np.ndarray, index: int
    ):
        """
        Get tokens from onset_notes and offset_notes, given the time index.
        """
        if len(onset_notes) > 0:
            onset_tokens = [1 + self.velocity_token_offset] + [
                _note_to_token(note, pitch_token_offset=self.pitch_token_offset)
                for note in onset_notes
            ]
        else:
            onset_tokens = []
        if len(offset_notes) > 0:
            offset_tokens = [0 + self.velocity_token_offset] + [
                _note_to_token(note, pitch_token_offset=self.pitch_token_offset)
                for note in offset_notes
            ]
        else:
            offset_tokens = []
        return [index + self.time_token_offset] + onset_tokens + offset_tokens


def notes_to_midi(notes: np.ndarray, beatstep: np.ndarray, offset_sec: float = 0.0):
    new_pm = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
    new_inst = pretty_midi.Instrument(program=0)

    new_inst.notes = [
        pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=beatstep[onset_idx] - offset_sec,
            end=beatstep[start_time] - offset_sec,
        )
        for onset_idx, start_time, pitch, velocity in notes
    ]
    new_pm.instruments.append(new_inst)
    new_pm.remove_invalid_notes()

    return new_pm


def get_onset_notes(notes: np.ndarray, index: int):
    return notes[notes[:, 0] == index]


def get_offset_notes(notes: np.ndarray, index: int):
    return notes[notes[:, 1] == index]


@njit
def _note_to_token(
    note: np.ndarray,
    pitch_token_offset: int,
) -> int:
    onset_idx, start_time, pitch, velocity = note
    return pitch + pitch_token_offset


@njit
def _tokens_to_note(
    notes: np.ndarray,
    onset_time_index: int,
    pitch: int,
    velocity: int,
) -> np.ndarray:
    # note onset
    # the offset time is set to -1 as dummy placement
    if velocity > 0:
        append_note = np.int_([[onset_time_index, -1, pitch, velocity]])
        notes = np.vstack((notes, append_note))

    # note offset
    elif velocity == 0:
        offset_note_idx = np.where(
            (
                (notes[:, 0] < onset_time_index)
                & (notes[:, 1] == -1)
                & (notes[:, 2] == pitch)
            )
        )
        # ignore note is onset is not found
        if len(offset_note_idx) > 0:
            notes[offset_note_idx[0], 1] = onset_time_index

    return notes
