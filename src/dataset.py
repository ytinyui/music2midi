import multiprocessing
from pathlib import Path
from typing import Literal, NamedTuple

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import T5Config

from .tokenizer import MidiTokenizer

PAD = T5Config.from_pretrained("t5-small").pad_token_id


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


class InputDataTuple(NamedTuple):
    inputs_embeds: torch.Tensor
    labels: torch.Tensor
    genre_id: torch.LongTensor
    difficulty_id: torch.LongTensor


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, config_path: str = "config.yaml"):
        super().__init__()
        self.config_path = config_path
        self.config = OmegaConf.load(self.config_path)
        self.data_dir = Path(data_dir)
        self.dataset_split_ids = np.load(
            self.data_dir / "dataset_split.npz", allow_pickle=True
        )

    def setup(self, stage=None):
        self.train_set = MyDataset(
            self.data_dir,
            self.dataset_split_ids.get("train_id"),
            self.config_path,
        )
        self.val_set = MyDataset(
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


class MyDataset(Dataset):
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
        self.tokenizer = MidiTokenizer(self.config)

        self.audio_paths = [
            str(data_dir / "audio" / f"{score_id}.wav") for score_id in score_ids
        ]
        if self.tokenizer.time_step:
            self.midi_notes = [
                np.load(data_dir / "midi_numpy" / f"{score_id}.npy")
                for score_id in score_ids
            ]
        else:
            self.midi_notes = [
                np.load(data_dir / "midi_quantized_numpy" / f"{score_id}.npy")
                for score_id in score_ids
            ]
            self.beat_times_interpolated = [
                np.load(data_dir / "beat_times_interpolated" / f"{score_id}.npy")
                for score_id in score_ids
            ]
        self.beat_times = [
            np.load(data_dir / "beat_times" / f"{score_id}.npy")
            for score_id in score_ids
        ]
        self.genre_ids = [
            self.metadata_dict.get("genre", score_id) for score_id in score_ids
        ]
        self.difficulty_ids = [
            self.metadata_dict.get("difficulty", score_id) for score_id in score_ids
        ]

    def __getitem__(self, index) -> InputDataTuple:
        segment_duration = self.config.dataset.segment_duration
        max_num_tokens = (
            self.config.dataset.max_num_tokens_per_second * segment_duration
        )
        max_beat_times_fluctuation = self.config.dataset.max_beat_times_fluctuation
        beat_times = self.beat_times[index]
        genre_id = self.genre_ids[index]
        difficulty_id = self.difficulty_ids[index]

        audio_path = self.audio_paths[index]
        full_duration = librosa.get_duration(filename=audio_path)

        while True:
            if self.tokenizer.time_step:
                start_time = np.random.choice(
                    np.arange(
                        0, full_duration - segment_duration, self.tokenizer.time_step
                    )
                )
                beat_times_segment = beat_times[
                    (beat_times >= start_time)
                    & (beat_times < start_time + segment_duration)
                ]
                notes_segment = get_notes_segment(
                    self.midi_notes[index], start_time, start_time + segment_duration
                )
                tokens = self.tokenizer(
                    notes_segment, start_time=start_time, add_eos=True
                )
            else:
                beat_times_interpolated = self.beat_times_interpolated[index]
                start_time = np.random.choice(
                    beat_times[beat_times <= full_duration - segment_duration]
                )
                beat_times_segment = beat_times[
                    (beat_times >= start_time)
                    & (beat_times < start_time + segment_duration)
                ]
                beat_times_interpolated_segment = beat_times_interpolated[
                    (beat_times_interpolated >= start_time)
                    & (beat_times_interpolated < start_time + segment_duration)
                ]

                start_index = np.searchsorted(beat_times_interpolated, start_time)
                end_index = start_index + len(beat_times_interpolated_segment) - 1
                notes_segment = get_notes_segment(
                    self.midi_notes[index], start_index, end_index
                )
                tokens = self.tokenizer(
                    notes_segment, start_time=start_index, add_eos=True
                )

            if len(beat_times_segment) < 3:
                beat_times_fluctuation = 0
            else:
                beat_times_fluctuation = np.diff(np.diff(beat_times_segment))
            if (
                len(tokens) <= max_num_tokens
                and np.max(np.abs(beat_times_fluctuation)) <= max_beat_times_fluctuation
            ):
                break
        waveform, sr = librosa.load(
            self.audio_paths[index],
            sr=self.config.dataset.sample_rate,
            offset=start_time,
            duration=segment_duration,
        )

        return InputDataTuple(
            inputs_embeds=torch.from_numpy(waveform),
            labels=torch.from_numpy(tokens),
            genre_id=torch.LongTensor([genre_id]),
            difficulty_id=torch.LongTensor([difficulty_id]),
        )

    def __len__(self):
        return len(self.audio_paths)


def get_notes_segment(
    notes: np.ndarray, start_time: float, end_time: float
) -> np.ndarray:
    return notes[(notes[:, 0] >= start_time) & (notes[:, 0] < end_time)]


def collate_fn(batch):
    x_batch, y_batch, genre_id_batch, difficulty_id_batch = zip(*batch)
    x_batch = pad_sequence(x_batch, batch_first=True, padding_value=0)
    y_batch = pad_sequence(y_batch, batch_first=True, padding_value=PAD)
    genre_id_batch = torch.stack(genre_id_batch)
    difficulty_id_batch = torch.stack(difficulty_id_batch)
    return InputDataTuple(x_batch, y_batch, genre_id_batch, difficulty_id_batch)
