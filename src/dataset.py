import multiprocessing
from pathlib import Path

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from .input import ModelInputs


class MetadataDict:
    def __init__(self, piano_ids: list[str], data_dir: Path, config_path: str):
        super().__init__()
        self.data_dir = data_dir
        self.config = OmegaConf.load(config_path)
        self.key_dict = {
            key: {item: i for i, item in enumerate(self.config.conditioning.get(key))}
            for key in self.config.conditioning.keys()
        }

        def read_metadata_fn(piano_id):
            return piano_id, OmegaConf.load(
                self.data_dir / "metadata" / f"{piano_id}.yaml"
            )

        meta_list = Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
            delayed(read_metadata_fn)(piano_id) for piano_id in piano_ids
        )
        self.meta_dict = {k: v for k, v in meta_list}

    def get(self, piano_id: str) -> list[int]:
        """
        Get the conditioning index of the sample, given piano_id
        """
        return [v[self.meta_dict[piano_id].piano[k]] for k, v in self.key_dict.items()]


class Music2MIDIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, config_path: str = "config.yaml"):
        super().__init__()
        self.config_path = config_path
        self.config = OmegaConf.load(self.config_path)
        self.data_dir = Path(data_dir)
        self.dataset_split_ids = np.load(
            self.data_dir / "dataset_split.npz", allow_pickle=True
        )

    def setup(self, stage=None):
        self.train_set = Music2MIDIDataset(
            self.data_dir,
            self.dataset_split_ids.get("train_id"),
            self.config_path,
        )
        self.val_set = Music2MIDIDataset(
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


class Music2MIDIDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        piano_ids: list[str],
        config_path: Path,
    ):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        self.metadata_dict = MetadataDict(piano_ids, data_dir, config_path)
        self.data_dir = data_dir

        self.audio_paths = [
            str(data_dir / "audio" / f"{piano_id}.wav") for piano_id in piano_ids
        ]
        self.midi_notes = [
            np.load(data_dir / "midi_numpy" / f"{piano_id}.npy")
            for piano_id in piano_ids
        ]
        self.cond_indices = [self.metadata_dict.get(piano_id) for piano_id in piano_ids]

    def __getitem__(self, index) -> ModelInputs:
        segment_duration = self.config.dataset.segment_duration
        max_num_notes = self.config.dataset.max_notes_per_second * segment_duration
        conditioning = self.cond_indices[index]

        audio_path = self.audio_paths[index]
        full_duration = librosa.get_duration(path=audio_path)

        while True:
            start_time = np.random.choice(
                np.arange(0, full_duration - segment_duration, segment_duration)
            )
            end_time = start_time + segment_duration
            notes_segment = get_notes_segment(
                self.midi_notes[index],
                start_time,
                end_time,
                shift_to_start_time=True,
            )
            if 0 < len(notes_segment) <= max_num_notes:
                break

        waveform, sr = librosa.load(
            self.audio_paths[index],
            sr=self.config.dataset.sample_rate,
            offset=start_time,
            duration=segment_duration,
        )
        if np.random.rand() < 0.5:
            waveform = librosa.util.normalize(waveform)
        transpose_step = np.random.randint(-6, 6)
        waveform, notes_segment = transpose(waveform, notes_segment, transpose_step, sr)

        return (
            torch.from_numpy(waveform),
            notes_segment,
            [torch.LongTensor([index]) for index in conditioning],
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


def transpose(waveform, notes, step, sr):
    waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=step)
    notes[:, 2] += step
    return waveform, notes


def collate_fn(batch):
    waveform, notes_batch, cond_index = zip(*batch)
    waveform = torch.stack(waveform)
    cond_index = torch.Tensor(cond_index).long()  # (batch, n_cond)
    return ModelInputs(waveform, notes_batch, cond_index)
