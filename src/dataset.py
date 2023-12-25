import os
from pathlib import Path
from typing import Literal, NamedTuple

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import T5Config

from .tokenizer import TokenizerFactory

PAD = T5Config.from_pretrained("t5-small").pad_token_id


class MetadataDict:
    def __init__(self, data_dir: Path, config_path: str):
        super().__init__()
        self.data_dir = data_dir
        self.config = OmegaConf.load(config_path)
        self.key_dict = {
            "genre": {k: v for k, v in self.config.genre_id.items()},
            "difficulty": {k: v for k, v in self.config.difficulty_id.items()},
        }

    def get(self, key: Literal["genre", "difficulty"], score_id: str):
        """
        Get the genre or difficulty id based on score_id
        """
        meta = OmegaConf.load(self.data_dir / "metadata" / (score_id + ".yaml"))
        return self.key_dict[key][meta.score.get(key)]


class InputDataTuple(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
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
        self.metadata_dict = MetadataDict(data_dir, config_path)
        self.data_dir = data_dir
        self.tokenizer = TokenizerFactory.create_tokenizer(self.config.tokenizer)

        self.audio_paths = [
            os.path.join(data_dir, "audio", score_id + ".wav") for score_id in score_ids
        ]
        self.midi_notes = [
            np.load(os.path.join(data_dir, "midi_numpy", score_id + ".npy"))
            for score_id in score_ids
        ]
        self.genre_ids = [
            self.metadata_dict.get("genre", score_id) for score_id in score_ids
        ]
        self.difficulty_ids = [
            self.metadata_dict.get("difficulty", score_id) for score_id in score_ids
        ]

    def __getitem__(self, index) -> InputDataTuple:
        duration = self.config.dataset.segment_duration
        genre_id = self.genre_ids[index]
        difficulty_id = self.difficulty_ids[index]

        audio_path = self.audio_paths[index]
        audio_duration = librosa.get_duration(filename=audio_path)
        start_time = np.random.choice(
            np.arange(0, audio_duration - duration, self.tokenizer.time_step)
        )
        waveform, sr = librosa.load(
            self.audio_paths[index],
            sr=self.config.dataset.sample_rate,
            offset=start_time,
            duration=duration,
        )
        notes_segment = get_notes_segment(
            self.midi_notes[index], start_time, start_time + duration
        )
        tokens = self.tokenizer(notes_segment, start_time=start_time, add_eos=True)

        return InputDataTuple(
            x=torch.from_numpy(waveform),
            y=torch.from_numpy(tokens),
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
    x_batch = pad_sequence(x_batch, batch_first=True, padding_value=PAD)
    y_batch = pad_sequence(y_batch, batch_first=True, padding_value=PAD)
    genre_id_batch = torch.stack(genre_id_batch)
    difficulty_id_batch = torch.stack(difficulty_id_batch)

    return InputDataTuple(x_batch, y_batch, genre_id_batch, difficulty_id_batch)
