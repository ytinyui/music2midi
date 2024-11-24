from __future__ import annotations

from typing import Iterable, Literal, Optional, Union

import numpy as np
import torch
from numba import njit
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

PAD = 0
BOS = 1
EOS = 2
ONSET = 3
OFFSET = 4


class MidiTokenizer:
    def __init__(self, config: DictConfig):
        self.config = config.tokenizer

        self.time_step = self.config.midi_quantize_ms / 1000
        self.pitch_token_offset = self.config.vocab_size.special
        self.time_token_offset = self.pitch_token_offset + self.config.vocab_size.pitch

    def to_string(self, tokens: np.ndarray) -> list[str]:
        def _to_string(token: int) -> str:
            if token == PAD:
                return "PAD"
            if token == BOS:
                return "BOS"
            if token == EOS:
                return "EOS"
            if token == ONSET:
                return "ONSET"
            if token == OFFSET:
                return "OFFSET"
            if token >= self.time_token_offset:
                return f"time_{token - self.time_token_offset}"
            if token >= self.pitch_token_offset:
                return f"note_{token - self.pitch_token_offset}"
            raise ValueError(f"Invalid token '{token}'")

        return [_to_string(token) for token in tokens]

    def decode(
        self,
        tokens_batch: Iterable[Union[np.ndarray, torch.Tensor]],
        mode: Literal["batched", "sequential"] = "batched",
        duration_per_batch: Optional[float] = None,
        cutoff_time: Optional[int] = None,
    ) -> Union[list[np.ndarray], np.ndarray]:
        """
        Use this method only if second is time unit

        input: list of tokens
        order: (batch, seq_len)
            - example: input[0  , 0], input[0, 1], ..., input[0, L-1],
                       input[1  , 0], input[1, 1], ..., input[1, L-1],
                       ...
                       input[N-1, 0], input[N, 1], ..., input[N, L-1]

        if mode == "batched", tokens in the batch are decoded independently (return: list[np.ndarray])
        if mode == "sequential", tokens in the batch are decoded in a single sequence (return: (L',))

        cutoff_time: optional. Drop all notes after cutoff_time
        duration_per_batch: duration of each batch (in seconds)
        """
        if mode == "batched":
            return [self._decode(tokens, 0, cutoff_time) for tokens in tokens_batch]
        elif mode == "sequential":
            assert (
                duration_per_batch is not None
            ), 'duration_per_batch is required for mode="sequential"'
            ret = []
            start_idx = 0
            n_steps = round(duration_per_batch / self.time_step)
            for tokens in tokens_batch:
                notes = self._decode(tokens, start_idx, cutoff_time)
                ret.append(notes)
                start_idx += n_steps

            return np.concatenate(ret)
        raise ValueError(f"Invalid argument mode={mode}")

    def __call__(
        self,
        notes_batch: Iterable[np.ndarray],
        cutoff_time: Optional[int] = None,
    ) -> torch.Tensor:
        """
        tokenize a tuple of notes and return a batched tensor
        """
        assert isinstance(notes_batch, Iterable), "notes should be passed in batch"
        tokens_batch = [self._tokenize(notes, cutoff_time) for notes in notes_batch]
        return pad_sequence(tokens_batch, batch_first=True, padding_value=PAD).long()

    def _tokenize(
        self,
        notes: np.ndarray,
        cutoff_time: Optional[int] = None,
    ) -> torch.Tensor:
        """
        cutoff_time: optional. Drop all notes after cutoff_time

        notes[0]: (onset_time, offset_time, pitch, velocity)

        Notes:
            - EOS token is appended at the end of the tokens
            - if the relative time steps exceed vocab size, the time steps will be clipped.
        """
        if len(notes) == 0:
            tokens = []

        else:
            notes = np.copy(notes)
            if cutoff_time is not None:
                notes = notes[notes[:, 0] < cutoff_time]

            # min length of each note is 1 step
            notes[:, 1] = np.maximum(notes[:, 1], notes[:, 0] + self.time_step)
            # convert time to indices
            notes[:, :2] = notes[:, :2] / self.time_step
            notes[:, :2] = np.rint(np.nextafter(notes[:, :2], notes[:, :2] + 1))
            # clip time step values to fit vocab size
            notes[:, :2] = np.minimum(notes[:, :2], self.config.vocab_size.time - 1)

            time_indices = np.unique(notes[:, :2])
            tokens = [
                tok
                for index in time_indices
                for tok in self._get_tokens(
                    onset_notes=get_onset_notes(notes, index),
                    offset_notes=get_offset_notes(notes, index),
                    index=index,
                )
            ]

        tokens.append(EOS)

        return torch.Tensor(tokens).long()

    def _decode(
        self,
        tokens: Union[np.ndarray, torch.Tensor],
        start_idx: int = 0,
        cutoff_time: Optional[int] = None,
    ) -> np.ndarray:
        """
        Decode tokens into notes array.
        If cutoff_time is provided, the note events after cutoff_time are ignored.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        notes = self._decode_tokens(tokens, start_idx)
        # remove notes without offset time
        notes = notes[notes[:, 1] != -1]
        # convert time index to seconds
        notes[:, :2] = notes[:, :2] * self.time_step

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
        cur_note_on = -1
        cur_note = -1

        for token in tokens:
            if token == EOS:
                break
            if token in [BOS, PAD]:
                continue
            if token == ONSET:
                cur_note_on = 1
            if token == OFFSET:
                cur_note_on = 0
            if token >= self.time_token_offset:
                cur_time_idx = start_idx + token - self.time_token_offset
                cur_note_on = -1
                cur_note = -1
            elif token >= self.pitch_token_offset:
                cur_note = token - self.pitch_token_offset

            if -1 in [cur_time_idx, cur_note_on, cur_note]:
                continue
            cur_velocity = cur_note_on * self.config.default_velocity
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
            onset_tokens = [ONSET] + [
                _note_to_token(note, pitch_token_offset=self.pitch_token_offset)
                for note in onset_notes
            ]
        else:
            onset_tokens = []
        if len(offset_notes) > 0:
            offset_tokens = [OFFSET] + [
                _note_to_token(note, pitch_token_offset=self.pitch_token_offset)
                for note in offset_notes
            ]
        else:
            offset_tokens = []
        return [index + self.time_token_offset] + onset_tokens + offset_tokens


def get_onset_notes(notes: np.ndarray, index: int):
    return notes[notes[:, 0] == index]


def get_offset_notes(notes: np.ndarray, index: int):
    return notes[notes[:, 1] == index]


@njit
def _note_to_token(
    note: np.ndarray,
    pitch_token_offset: int,
) -> int:
    onset_idx, offset_idx, pitch, velocity = note
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
    if velocity:
        append_note = np.int_([[onset_time_index, -1, pitch, velocity]])
        notes = np.vstack((notes, append_note))
    # note offset
    else:
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
