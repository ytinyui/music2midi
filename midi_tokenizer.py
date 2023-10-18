from __future__ import annotations

from typing import Optional

import numpy as np
import pretty_midi
from numba import njit
from omegaconf import DictConfig

PAD = 0
EOS = 1


class TokenizerFactory:
    @staticmethod
    def create_tokenizer(config: DictConfig) -> MidiTokenizerNoVelocity | MidiTokenizer:
        """
        Use this function to get the tokenizer.

        Example:
        config = OmegaConf.load("config.yaml")
        tokenizer = TokenizerFactory(config.tokenizer)
        """
        if config.no_velocity == True:
            return MidiTokenizerNoVelocity(config)
        else:
            return MidiTokenizer(config)


class TokenizerBase:
    """
    This is the base class of the MIDI Tokenizer.
    MIDITokenizer tokenizes note velocity while MIDITokenizerNoVelocity treats velocity as onsets.
    """

    def __init__(self, config: DictConfig):
        self.config = config
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
        cutoff_duration: Optional[int] = None,
    ) -> np.ndarray:
        """
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
            ret.append(
                self.decode(
                    tokens, start_idx=start_idx, cutoff_duration=cutoff_duration
                )
            )
            start_idx += n_steps

        return np.concatenate(ret)

    def __call__(
        self,
        notes: np.ndarray,
        start_time: Optional[int] = None,
        cutoff_duration: Optional[int] = None,
        add_eos: Optional[bool] = True,
    ) -> np.ndarray:
        """
        start_time: optional. Default: the time of the first onset
        cutoff_duration: optional. Drop all notes after cutoff_duration

        notes[0]: (onset_time, offset_time, pitch, velocity)

        Notes:
            - if start time is None, the time is aligned with the first onset
            - if add_eos is True, add EOS token at the end of the tokens list
            - if the relative time steps exceed vocab size, the time steps will be clipped.
        """
        if len(notes) == 0:
            tokens = np.array([])

        else:
            if start_time is None:
                start_time = notes[0, 0]
            notes[:, :2] -= start_time

            if cutoff_duration is not None:
                notes = notes[notes[:, 0] < cutoff_duration]

            # min length of each note is 1 step
            notes[:, 1] = np.maximum(notes[:, 1], notes[:, 0] + self.time_step)
            # convert time to steps
            notes[:, :2] = notes[:, :2] / self.time_step
            notes[:, :2] = np.rint(np.nextafter(notes[:, :2], notes[:, :2] + 1))

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
            # absolute time indices to relative time steps
            relative_time_steps = np.diff(time_indices, prepend=0)
            if np.sum(relative_time_steps >= self.config.vocab_size.time):
                # clip time step values
                relative_time_steps = np.minimum(
                    relative_time_steps, self.config.vocab_size.time - 1
                )
            tokens[tokens >= self.time_token_offset] = (
                relative_time_steps + self.time_token_offset
            )

        if add_eos:
            tokens = np.append(tokens, EOS)

        tokens = np.array(tokens, dtype=np.int_)

        return tokens

    def decode(
        self,
        tokens: np.ndarray,
        start_idx: Optional[int] = 0,
        cutoff_duration: Optional[int] = None,
    ) -> np.ndarray:
        """
        Decode tokens into notes array.
        If cutoff_duration is provided, the note events after cutoff_duration are ignored.
        """
        notes = self._decode_tokens(tokens, start_idx)
        # remove dummy element and notes without offset
        notes = notes[notes[:, 1] != -1]
        # convert note time
        notes[:, :2] = notes[:, :2] * self.time_step

        if cutoff_duration is not None:
            # drop note onsets beyond cutoff_duration
            notes = notes[notes[:, 0] < cutoff_duration]
            # truncate offsets
            notes[:, 1] = np.where(
                notes[:, 1] > cutoff_duration, cutoff_duration, notes[:, 1]
            )

        return notes

    def _decode_tokens(self, tokens: np.ndarray, start_idx: int) -> np.ndarray:
        """
        Decode tokens to numpy array (pretty_midi format)
        """
        notes = np.array([[-1, -1, -1, -1]])  # dummy element
        cur_time_idx = start_idx
        cur_velocity = -1
        cur_note = -1

        for token in tokens:
            if token == EOS:
                break
            if token >= self.time_token_offset:
                cur_time_idx += token - self.time_token_offset
                cur_velocity = -1
                cur_note = -1
            elif token >= self.velocity_token_offset:
                cur_velocity = token - self.velocity_token_offset
                if self.config.no_velocity:
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
            if not self.config.no_velocity:
                cur_velocity = -1

        return notes.astype(np.float_)

    def _get_tokens(
        self, onset_notes: np.ndarray, offset_notes: np.ndarray, index: int
    ):
        """
        Get tokens from onset_notes and offset_notes, given the time index.
        """
        return NotImplementedError


class MidiTokenizer(TokenizerBase):
    def __init__(self, config):
        super().__init__(config)

    def _get_tokens(
        self, onset_notes: np.ndarray, offset_notes: np.ndarray, index: int
    ):
        return (
            [index + self.time_token_offset]
            + [
                tok
                for note in onset_notes
                for tok in _tokenize(
                    note,
                    onset=True,
                    pitch_token_offset=self.pitch_token_offset,
                    velocity_token_offset=self.velocity_token_offset,
                )
            ]
            + [
                tok
                for note in offset_notes
                for tok in _tokenize(
                    note,
                    onset=False,
                    pitch_token_offset=self.pitch_token_offset,
                    velocity_token_offset=self.velocity_token_offset,
                )
            ]
        )


class MidiTokenizerNoVelocity(TokenizerBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _get_tokens(
        self, onset_notes: np.ndarray, offset_notes: np.ndarray, index: int
    ):
        onset_tokens = (
            [1 + self.velocity_token_offset]
            + [
                _tokenize_no_velocity(note, pitch_token_offset=self.pitch_token_offset)
                for note in onset_notes
            ]
            if len(onset_notes) > 0
            else []
        )
        offset_tokens = (
            [0 + self.velocity_token_offset]
            + [
                _tokenize_no_velocity(note, pitch_token_offset=self.pitch_token_offset)
                for note in offset_notes
            ]
            if len(offset_notes) > 0
            else []
        )
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
def _tokenize(
    note: np.ndarray,
    onset: bool,
    pitch_token_offset: int,
    velocity_token_offset: int,
) -> list:
    onset_idx, start_time, pitch, velocity = note
    if onset:
        return [velocity + velocity_token_offset, pitch + pitch_token_offset]
    else:
        return [velocity_token_offset, pitch + pitch_token_offset]


@njit
def _tokenize_no_velocity(
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
