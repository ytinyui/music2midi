import multiprocessing
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from .input import ModelInputs
from .utils import numpy_to_midi, pad_audio


class MetadataDict:
    def __init__(self, score_ids: list[str], data_dir: Path, config_path: str):
        super().__init__()
        self.data_dir = data_dir
        self.config = OmegaConf.load(config_path)
        self.key_dict = {
            "genre": {k: v for k, v in self.config.genre_id.items()},
            "difficulty": {k: v for k, v in self.config.difficulty_id.items()},
        }
        load_meta_fn = lambda x: (
            x,
            OmegaConf.load(self.data_dir / "metadata" / f"{x}.yaml"),
        )
        meta_list = Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
            delayed(load_meta_fn)(score_id) for score_id in score_ids
        )
        self.meta_dict = {k: v for k, v in meta_list}

    def get(self, key: Literal["genre", "difficulty"], score_id: str):
        """
        Get the genre or difficulty id of the sample, given score_id
        """
        return self.key_dict[key][self.meta_dict[score_id].score[key]]


class Music2MidiDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, config_path: str = "config.yaml"):
        super().__init__()
        self.config_path = config_path
        self.config = OmegaConf.load(self.config_path)
        self.data_dir = Path(data_dir)
        self.dataset_split_ids = np.load(
            self.data_dir / "dataset_split.npz", allow_pickle=True
        )

    def setup(self, stage=None):
        self.train_set = Music2MidiDataset(
            self.data_dir,
            self.dataset_split_ids.get("train_id"),
            self.config_path,
        )
        self.val_set = Music2MidiDataset(
            self.data_dir,
            self.dataset_split_ids.get("val_id"),
            self.config_path,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            collate_fn=collate_fn,
            **self.config.dataloader,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            shuffle=False,
            collate_fn=collate_fn,
            **self.config.dataloader,
        )


class Music2MidiDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        score_ids: list[str],
        config_path: Path,
    ):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        self.metadata_dict = MetadataDict(score_ids, data_dir, config_path)
        self.data_dir = data_dir

        self.audio_paths = [
            str(data_dir / "audio" / f"{score_id}.wav") for score_id in score_ids
        ]
        self.beat_times_aligned = [
            np.load(data_dir / "beat_times_aligned" / f"{score_id}.npy")
            for score_id in score_ids
        ]
        if self.config.dataset.quantize_sub_beats:
            self.midi_notes = [
                np.load(data_dir / "midi_quantized_numpy" / f"{score_id}.npy")
                for score_id in score_ids
            ]
            self.beat_times_midi_interp = [
                np.load(data_dir / "beat_times_midi_interp" / f"{score_id}.npy")
                for score_id in score_ids
            ]
            self.warp_paths = [
                np.load(data_dir / "warp_path" / f"{score_id}.npy")
                for score_id in score_ids
            ]
        else:
            self.midi_notes = [
                np.load(data_dir / "midi_numpy" / f"{score_id}.npy")
                for score_id in score_ids
            ]
        self.genre_ids = [
            self.metadata_dict.get("genre", score_id) for score_id in score_ids
        ]
        self.difficulty_ids = [
            self.metadata_dict.get("difficulty", score_id) for score_id in score_ids
        ]

    def __getitem__(self, index) -> ModelInputs:
        segment_duration = self.config.dataset.segment_duration  #! change this
        segment_num_sub_beats = self.config.dataset.segment_num_sub_beats
        max_beat_times_fluctuation = self.config.dataset.max_beat_times_fluctuation
        beat_times_aligned = self.beat_times_aligned[index]
        genre_id = self.genre_ids[index]
        difficulty_id = self.difficulty_ids[index]

        audio_path = self.audio_paths[index]
        full_duration = librosa.get_duration(path=audio_path)

        while True:
            if self.config.dataset.quantize_sub_beats:
                # midi note time is beat index
                warp_path = self.warp_paths[index]
                beat_times_midi = self.beat_times_midi_interp[index]
                midi_start_index = np.random.randint(
                    0, len(beat_times_midi) - 1 - segment_num_sub_beats
                )
                midi_end_index = midi_start_index + segment_num_sub_beats
                notes_segment = get_notes_segment(
                    self.midi_notes[index],
                    midi_start_index,
                    midi_end_index,
                    shift_to_start_time=True,
                )
                midi_start_time = beat_times_midi[midi_start_index]
                midi_end_time = beat_times_midi[midi_end_index]
                # audio start time
                start_time = np.interp(midi_start_time, warp_path[1], warp_path[0])
                end_time = np.interp(midi_end_time, warp_path[1], warp_path[0])
                # ? segment_duration is different across batch, implement without num_sub_beats
                segment_duration = end_time - start_time
                beat_times_segment = beat_times_aligned[
                    (beat_times_aligned >= start_time) & (beat_times_aligned < end_time)
                ]
            else:
                start_time = np.random.choice(
                    np.arange(0, full_duration - segment_duration, segment_duration)
                )
                end_time = start_time + segment_duration
                beat_times_segment = beat_times_aligned[
                    (beat_times_aligned >= start_time) & (beat_times_aligned < end_time)
                ]
                notes_segment = get_notes_segment(
                    self.midi_notes[index],
                    start_time,
                    end_time,
                    shift_to_start_time=True,
                )
            if len(beat_times_segment) < 3:
                beat_times_fluctuation = 0
            else:
                beat_times_fluctuation = np.diff(np.diff(beat_times_segment))
            if np.max(np.abs(beat_times_fluctuation)) <= max_beat_times_fluctuation:
                break
        waveform, sr = librosa.load(
            self.audio_paths[index],
            sr=self.config.dataset.sample_rate,
            offset=start_time,
            duration=segment_duration,
        )
        notes_waveform = numpy_to_midi(notes_segment).synthesize(fs=sr)
        waveform, notes_waveform = pad_audio(waveform, notes_waveform, mode="fix_x")

        return (
            torch.from_numpy(waveform),
            notes_segment,
            torch.from_numpy(notes_waveform),
            torch.LongTensor([genre_id]),
            torch.LongTensor([difficulty_id]),
        )

    def __len__(self):
        return len(self.audio_paths)


def get_notes_segment(
    notes: np.ndarray,
    start_time: float,
    end_time: float,
    shift_to_start_time=False,
) -> np.ndarray:
    ret = notes[(notes[:, 0] >= start_time) & (notes[:, 0] < end_time)]
    if shift_to_start_time:
        ret[:, :2] -= start_time
    return ret


def collate_fn(batch):
    waveform, notes_batch, notes_waveform, genre_id, difficulty_id = zip(*batch)
    waveform = torch.stack(waveform)
    notes_waveform = torch.stack(notes_waveform)
    genre_id = torch.stack(genre_id)
    difficulty_id = torch.stack(difficulty_id)
    return ModelInputs(waveform, notes_batch, notes_waveform, genre_id, difficulty_id)
