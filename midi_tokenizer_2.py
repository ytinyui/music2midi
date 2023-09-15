import numpy as np
from numba import njit


PAD = 0
EOS = 1
RESERVED_1 = 2
RESERVED_2 = 3


class MidiTokenizer:
    def __init__(self, config):
        self.config = config

        self._pitch_offset_idx = self.config.vocab_size.special
        self._velocity_offset_idx = (
            self._pitch_offset_idx + self.config.vocab_size.pitch
        )
        self._time_offset_idx = (
            self._velocity_offset_idx + self.config.vocab_size.velocity
        )

    def _tokenize(self, note: np.ndarray, onset: bool) -> list:
        """
        tokenize a note
        onset = True: return onset tokens: [pitch, velocity]
        onset = False: set velocity to zero
        """
        return _fast_tokenize(
            note, onset, self._pitch_offset_idx, self._velocity_offset_idx
        )

    def to_string(self, tokens: np.ndarray) -> list[str]:
        def _to_string(token: int) -> str:
            if token == PAD:
                return "PAD"
            if token == EOS:
                return "EOS"
            if token == RESERVED_1:
                return "RESERVED_1"
            if token == RESERVED_2:
                return "RESERVED_2"
            if token >= self._time_offset_idx:
                return ("time", token - self._time_offset_idx)
            if token >= self._velocity_offset_idx:
                return ("velocity", token - self._velocity_offset_idx)
            if token >= self._pitch_offset_idx:
                return ("pitch", token - self._pitch_offset_idx)

            raise ValueError(f"Invalid token '{token}'")

        return [_to_string(token) for token in tokens]

    def __call__(
        self,
        notes: np.ndarray,
        time_offset: int = None,
        add_eos: bool = True,
        time_cutoff: int = None,
    ) -> np.ndarray:
        """
        - note[0]: (onset_idx, offset_idx, pitch, velocity)
        - time_offset: optional, if time_offset is None, the time is aligned with the first onset
        - time_cutoff: optional, drop all notes after time_cutoff

        - return format: [time 0, pitch, velocity, pitch, velocity, time 1, pitch, velocity, time 2, ...]

        Notes:
            - if time_offset is None, the time is aligned with the first onset
            - if add_eos is True, add EOS token at the end of the tokens list
            - if time_cutoff is None, notes with times beyond the vocab size for time are dropped
            - time_offset is applied before time_cutoff

        Example:
            notes = np.array([[0, 1, 60, 80], [0, 2, 55, 90], [1, 3, 40, 100]])
            tokens = tokenizer(notes)
            tokens
            # array([258,  62, 210,  57, 220, 259,  42, 230,  62, 130, 260,  57, 130,
                     261,  42, 130,   1])
        """

        def get_onset_notes(notes: np.ndarray, time_idx: int) -> np.ndarray:
            return notes[notes[:, 0] == time_idx]

        def get_offset_notes(notes: np.ndarray, time_idx: int) -> list:
            return list(filter(lambda x: x[1] == time_idx, notes))

        if time_offset is None:
            time_offset = notes[0, 0]
        notes[:, :2] -= time_offset

        if time_cutoff is None:
            time_cutoff = self.config.vocab_size.time

        time_indices = np.unique(notes[:, :2])
        time_indices = time_indices[time_indices < time_cutoff]

        tokens = []
        for time_idx in time_indices:
            onset_notes = notes[notes[:, 0] == time_idx]
            offset_notes = notes[notes[:, 1] == time_idx]

            onset_tokens = [
                token
                for note in onset_notes
                for token in self._tokenize(note, onset=True)
            ]
            offset_tokens = [
                token
                for note in offset_notes
                for token in self._tokenize(note, onset=False)
            ]
            # add beatstep token before notes token
            tokens += [time_idx + self._time_offset_idx] + onset_tokens + offset_tokens

        if add_eos:
            tokens.append(EOS)
        return np.array(tokens, dtype=np.int_)

    def decode(
        self, tokens: np.ndarray, time_offset: int = None, time_cutoff: int = None
    ) -> np.ndarray:
        notes = np.array([[-1, -1, -1, -1]])  # dummy element
        cur_time = -1
        cur_velocity = -1
        cur_note = -1

        for token in tokens:
            if token == EOS:
                break
            if token in [PAD, RESERVED_1, RESERVED_2]:
                continue
            if token >= self._time_offset_idx:
                cur_time = token - self._time_offset_idx
            elif token >= self._velocity_offset_idx:
                cur_velocity = token - self._velocity_offset_idx
            elif token >= self._pitch_offset_idx:
                cur_note = token - self._pitch_offset_idx
            if -1 in [cur_time, cur_velocity, cur_note]:
                continue

            # note onset
            if cur_velocity > 0:
                append_note = np.array(
                    [[cur_time, -1, cur_note, cur_velocity]], dtype=np.int_
                )
                notes = np.vstack([notes, append_note])

            # note offset
            elif cur_velocity == 0:
                note_idx = np.where(
                    (
                        (notes[:, 0] < cur_time)
                        & (notes[:, 1] == -1)
                        & (notes[:, 2] == cur_note)
                    )
                )
                notes[note_idx, 1] = cur_time

            cur_velocity = -1
            cur_note = -1

        # remove dummy element and notes without offset
        notes = notes[notes[:, 1] != -1]

        if time_offset is not None:
            notes[:, :2] -= time_offset

        if time_cutoff is not None:
            notes = notes[(notes[:, 0] < time_cutoff) & (notes[:, 1] < time_cutoff)]

        return notes


@njit(cache=True)
def _fast_tokenize(
    note: np.ndarray, onset: bool, pitch_offset, velocity_offset
) -> list:
    onset_idx, offset_idx, pitch, velocity = note
    if onset:
        return [pitch + pitch_offset, velocity + velocity_offset]
    else:
        return [pitch + pitch_offset, velocity_offset]
