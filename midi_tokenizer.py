from __future__ import annotations
import numpy as np
from numba import njit
import pretty_midi
from omegaconf import DictConfig


PAD = 0
EOS = 1


def get_tokenizer(config) -> MidiTokenizerNoVelocity | MidiTokenizer:
    """
    Use this function to get the tokenizer.
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
        self.mask_offset = self.config.vocab_size.special
        self.pitch_offset = self.mask_offset + self.config.vocab_size.mask
        self.velocity_offset = self.pitch_offset + self.config.vocab_size.pitch
        self.time_offset = self.velocity_offset + self.config.vocab_size.velocity

    def to_string(self, tokens: np.ndarray) -> list[str]:
        def _to_string(token: int) -> str:
            if token == PAD:
                return "PAD"
            if token == EOS:
                return "EOS"
            if token >= self.time_offset:
                return ("time", token - self.time_offset)
            if token >= self.velocity_offset:
                return ("velocity", token - self.velocity_offset)
            if token >= self.pitch_offset:
                return ("note", token - self.pitch_offset)
            if token >= self.mask_offset:
                return ("mask", token - self.mask_offset)

            raise ValueError(f"Invalid token '{token}'")

        return [_to_string(token) for token in tokens]

    def batch_decode(
        self, batched_tokens: np.ndarray, n_steps: int, cutoff_idx: int = None
    ) -> np.ndarray:
        """
        input: [N, L]
        order: batch then seq_len
            - example: input[0  , 0], input[0, 1], ..., input[0, L-1],
                       input[1  , 0], input[1, 1], ..., input[1, L-1],
                       ...
                       input[N-1, 0], input[N, 1], ..., input[N, L-1]

        n_steps: step offset per batch
        return: [L']
        """

        ret = []
        start_idx = 0
        for tokens in batched_tokens:
            ret.append(self.decode(tokens, start_idx=start_idx, cutoff_idx=cutoff_idx))
            start_idx += n_steps

        return np.concatenate(ret)

    def __call__(
        self,
        notes: np.ndarray,
        offset_idx: int = 0,
        add_eos: bool = True,
        cutoff_idx: int = None,
    ) -> np.ndarray:
        """
        - note[0]: (onset_idx, offset_idx, pitch, velocity)
        - offset_idx: optional, if offset_idx is -1, the time is aligned with the first onset
        - cutoff_idx: optional, drop all notes after cutoff_idx

        - return format: [onset time 1, pitch 1, velocity 1, offset time 1, pitch 1, 0,
                          onset time 2, pitch 2, velocity 2, offset time 2, ...]

        Notes:
            - if offset_idx is None, the time is aligned with the first onset
            - if add_eos is True, add EOS token at the end of the tokens list
            - if cutoff_idx is None, notes with times beyond the vocab size for time are dropped
            - offset_idx is applied before cutoff_idx
        """
        if len(notes) == 0:
            tokens = np.array([])

        else:
            if offset_idx == -1:
                offset_idx = notes[0, 0]
            notes[:, :2] -= offset_idx

            if cutoff_idx is None:
                cutoff_idx = self.config.vocab_size.time
            notes = notes[notes[:, 0] < cutoff_idx]
            notes[:, 1] = np.where(notes[:, 1] > cutoff_idx, cutoff_idx, notes[:, 1])

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
            if self.config.relative_time_steps:
                relative_time_steps = np.diff(time_indices, prepend=0)
                tokens[tokens >= self.time_offset] = (
                    relative_time_steps + self.time_offset
                )

        if add_eos:
            tokens = np.append(tokens, EOS)

        tokens = np.array(tokens, dtype=np.int_)

        return tokens

    def decode(
        self,
        tokens: np.ndarray,
        start_idx: int = 0,
        cutoff_idx: int = None,
    ) -> np.ndarray:
        """
        Decode tokens into notes array.
        If cutoff_idx is not None, the note events after cutoff_idx are ignored.
        """
        notes = self._decode_tokens(tokens, start_idx)
        # remove dummy element and notes without offset
        notes = notes[notes[:, 1] != -1]

        if cutoff_idx is not None:
            # drop note onsets beyond cutoff_idx
            notes = notes[notes[:, 0] < cutoff_idx]
            # truncate offsets
            notes[:, 1] = np.where(notes[:, 1] > cutoff_idx, cutoff_idx, notes[:, 1])

        return notes

    def _decode_tokens(self, tokens: np.ndarray, start_idx: int):
        notes = np.array([[-1, -1, -1, -1]])  # dummy element
        cur_time = start_idx
        cur_velocity = -1
        cur_note = -1

        for token in tokens:
            if token == EOS:
                break
            if token >= self.time_offset:
                if self.config.relative_time_steps:
                    cur_time += token - self.time_offset
                else:
                    cur_time = token - self.time_offset
                cur_velocity = -1
                cur_note = -1
            elif token >= self.velocity_offset:
                cur_velocity = token - self.velocity_offset
                if self.config.no_velocity:
                    cur_velocity *= self.config.default_velocity
                cur_note = -1
            elif token >= self.pitch_offset:
                cur_note = token - self.pitch_offset
            elif token >= self.mask_offset:
                continue

            if -1 in [cur_time, cur_velocity, cur_note]:
                continue

            notes = _tokens_to_note(notes, cur_time, cur_note, cur_velocity)

            cur_note = -1
            if not self.config.no_velocity:
                cur_velocity = -1

        return notes

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
            [index + self.time_offset]
            + [
                tok
                for note in onset_notes
                for tok in _tokenize(
                    note,
                    onset=True,
                    pitch_offset=self.pitch_offset,
                    velocity_offset=self.velocity_offset,
                )
            ]
            + [
                tok
                for note in offset_notes
                for tok in _tokenize(
                    note,
                    onset=False,
                    pitch_offset=self.pitch_offset,
                    velocity_offset=self.velocity_offset,
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
            [1 + self.velocity_offset]
            + [
                _tokenize_no_velocity(note, pitch_offset=self.pitch_offset)
                for note in onset_notes
            ]
            if len(onset_notes) > 0
            else []
        )
        offset_tokens = (
            [0 + self.velocity_offset]
            + [
                _tokenize_no_velocity(note, pitch_offset=self.pitch_offset)
                for note in offset_notes
            ]
            if len(offset_notes) > 0
            else []
        )
        return [index + self.time_offset] + onset_tokens + offset_tokens


def notes_to_midi(notes: np.ndarray, beatstep: np.ndarray, offset_sec: float = 0.0):
    new_pm = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
    new_inst = pretty_midi.Instrument(program=0)

    new_inst.notes = [
        pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=beatstep[onset_idx] - offset_sec,
            end=beatstep[offset_idx] - offset_sec,
        )
        for onset_idx, offset_idx, pitch, velocity in notes
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
    pitch_offset: int,
    velocity_offset: int,
) -> list:
    onset_idx, offset_idx, pitch, velocity = note
    if onset:
        return [velocity + velocity_offset, pitch + pitch_offset]
    else:
        return [velocity_offset, pitch + pitch_offset]


@njit
def _tokenize_no_velocity(
    note: np.ndarray,
    pitch_offset: int,
) -> int:
    onset_idx, offset_idx, pitch, velocity = note
    return pitch + pitch_offset


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
        append_note = np.array([[onset_time_index, -1, pitch, velocity]], dtype=np.int_)
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
            # print("velocity")
            notes[offset_note_idx[0], 1] = onset_time_index

    return notes
