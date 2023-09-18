import torch
import librosa
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from layer.input import LogMelSpectrogram
from midi_tokenizer import MidiTokenizerNoVelocity
from transformers import T5Config
from torch.nn.utils.rnn import pad_sequence


PAD = T5Config.from_pretrained("t5-small").pad_token_id
assert PAD == 0


class PopDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, config_path: str = "config.yaml"):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        self.data_dir = Path(data_dir)
        self.song_ids = [path.stem for path in self.data_dir.glob("*.yaml")]

    def setup(self, stage=None):
        train_ids, val_ids = train_test_split(self.song_ids, test_size=0.1)
        self.train_set = PopDataset(
            self.data_dir, train_ids, self.config, transform=True
        )
        self.val_set = PopDataset(self.data_dir, val_ids, self.config)

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
    def __init__(
        self,
        data_dir: Path,
        song_ids: list[str],
        config: DictConfig,
        transform: bool = False,
    ):
        super().__init__()

        self.config = config
        self.data_dir = data_dir
        self.transform = transform
        self.spectrogram = LogMelSpectrogram()
        self.tokenizer = MidiTokenizerNoVelocity(self.config.tokenizer)

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

        composer_dict = {}
        for k, v in config.composer_ids.items():
            composer_dict[k] = v
        self.composer_ids = [
            get_composer_id(str(self.data_dir / song_id) + ".yaml", composer_dict)
            for song_id in song_ids
        ]

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        n_bars = self.config.dataset.n_bars

        composer_id = torch.LongTensor([self.composer_ids[index]])

        beattimes = self.beattimes[index]
        downbeat_indices = [i for i, x in enumerate(beattimes) if i % 4 == 0][:-n_bars]

        start_idx = np.random.choice(downbeat_indices)
        notes_segment = get_notes_segment(
            self.midi_notes[index], start_idx, start_idx + n_bars * 8
        )

        waveform = get_audio_segment(
            self.audio_paths[index],
            self.config.dataset.sample_rate,
            beattimes,
            start_idx,
            start_idx + n_bars * 4,
        )

        # if self.transform:
        #     velocity_shift_val = np.random.binomial(10, p=0.5) - 5
        #     notes_segment = note_velocity_shift(notes_segment, value=velocity_shift_val)

        # x.shape = (length, feat_dim)
        x = self.spectrogram(waveform).transpose(-1, -2)
        tokens = self.tokenizer(notes_segment, offset_idx=start_idx, add_eos=True)
        y = torch.from_numpy(tokens)

        return x, y, composer_id

    def __len__(self):
        return len(self.audio_paths)


def get_composer_id(file_path: str, composer_dict: dict) -> int:
    return composer_dict[OmegaConf.load(file_path).piano.uploader]


def get_audio_segment(
    audio_path: Path,
    sample_rate: int,
    beattimes: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    start_time, end_time = beattimes[start_idx], beattimes[end_idx]
    waveform, sample_rate = librosa.load(
        str(audio_path),
        sr=sample_rate,
        offset=start_time,
        duration=end_time - start_time,
    )
    return torch.from_numpy(waveform)


def get_notes_segment(notes: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    return notes[(notes[:, 0] >= start_idx) & (notes[:, 0] < end_idx)]


def note_velocity_shift(notes: np.ndarray, value: int) -> np.ndarray:
    notes[:, 3] = np.minimum(np.maximum(notes[:, 3] + value, 1), 127)
    return notes


def collate_fn(batch):
    x_batch, y_batch, id_batch = zip(*batch)
    x_batch = pad_sequence(x_batch, batch_first=True, padding_value=PAD)
    y_batch = pad_sequence(y_batch, batch_first=True, padding_value=PAD)
    id_batch = torch.stack(id_batch)
    return x_batch, y_batch, id_batch
