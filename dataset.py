import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
from layer.input import LogMelSpectrogram
import librosa
from midi_tokenizer import MidiTokenizer
from transformers import T5Config
from torch.nn.utils.rnn import pad_sequence


PAD = T5Config.from_pretrained("t5-small").pad_token_id
assert PAD == 0


class PopDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, config_path: str = "config.yaml"):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        self.dataset = PopDataset(data_dir, config_path)

    def setup(self, stage=None):
        self.dataset.midi_notes = [
            truncate_notes(notes) for notes in self.dataset.midi_notes
        ]
        self.train_set, self.val_set = random_split(self.dataset, (0.9, 0.1))

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            collate_fn=collate_fn,
            **self.config.dataloader
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, collate_fn=collate_fn, **self.config.dataloader)


class PopDataset(Dataset):
    def __init__(self, data_dir: str, config_path: str = "config.yaml"):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = OmegaConf.load(config_path)
        self.spectrogram = LogMelSpectrogram()
        self.tokenizer = MidiTokenizer(self.config.tokenizer)

        song_ids = [p.stem for p in self.data_dir.glob("*.yaml")]
        self.audio_paths = [
            (self.data_dir / song_id).glob("*.pitchshift.wav").__next__()
            for song_id in song_ids
        ]

        self.midi_notes = [
            np.load((self.data_dir / song_id).glob("*notes.npy").__next__())
            for song_id in song_ids
        ]
        self.beattimes = [
            np.load((self.data_dir / song_id).glob("*beattime.npy").__next__())
            for song_id in song_ids
        ]

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        n_bars = self.config.dataset.n_bars
        # index = 1
        # print(self.audio_paths[index])
        beattimes = self.beattimes[index]
        downbeat_indices = [i for i, x in enumerate(beattimes) if i % 4 == 0][:-n_bars]
        # resample if list is empty
        while True:
            start_idx = np.random.choice(downbeat_indices)
            notes_segment = get_notes_segment(
                self.midi_notes[index], start_idx, start_idx + n_bars * 8
            )
            if len(notes_segment) > 0:
                break
        # print(beattimes[start_idx])
        audio = get_audio_segment(
            self.audio_paths[index],
            beattimes,
            start_idx,
            start_idx + n_bars * 4,
            self.config.dataset.sample_rate,
        )

        # x.shape = (length, feat_dim)
        x = self.spectrogram(torch.from_numpy(audio)).transpose(-1, -2)
        tokens = self.tokenizer.notes_to_relative_tokens(
            notes_segment, offset_idx=start_idx, add_eos=True
        )
        y = torch.from_numpy(tokens)

        return x, y

    def __len__(self):
        return len(self.audio_paths)


def get_audio_segment(
    audio_path: Path,
    beattimes: np.ndarray,
    start_idx: int,
    end_idx: int,
    sample_rate: int,
) -> np.ndarray:
    start_time, end_time = beattimes[start_idx], beattimes[end_idx]
    audio, sample_rate = librosa.load(
        str(audio_path),
        sr=sample_rate,
        offset=start_time,
        duration=end_time - start_time,
    )
    return audio


def get_notes_segment(notes: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    return notes[(notes[:, 0] >= start_idx) & (notes[:, 0] < end_idx)]


def truncate_notes(notes: np.ndarray, note_max_dur: int = 100) -> np.ndarray:
    for x in notes:
        x[1] = x[0] + min(x[1] - x[0], note_max_dur)
    return notes


def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    x_batch = pad_sequence(x_batch, batch_first=True, padding_value=PAD)
    y_batch = pad_sequence(y_batch, batch_first=True, padding_value=PAD)
    return x_batch, y_batch
