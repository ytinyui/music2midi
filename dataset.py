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

class PopDataset(Dataset):
    def __init__(self, data_dir: str, config_path:str="config.yaml"):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = OmegaConf.load(config_path)
        self.spectrogram = LogMelSpectrogram()
        self.tokenizer = MidiTokenizer(self.config.tokenizer)
        
        song_ids = [p.stem for p in self.data_dir.glob("*.yaml")]            
        self.audio_paths = [(self.data_dir / song_id).glob("*.pitchshift.wav").__next__() for song_id in song_ids]
        
        self.midi_notes = [np.load((self.data_dir / song_id).glob("*notes.npy").__next__()) for song_id in song_ids]
        self.beat_times = [np.load((self.data_dir / song_id).glob("*beattime.npy").__next__()) for song_id in song_ids]
        # self.beat_steps = [np.load((self.data_dir / song_id).glob("*beatstep.npy").__next__()) for song_id in song_ids]
        # self.beat_intervals = [np.load((self.data_dir / song_id).glob("*beatinterval.npy").__next__()) for song_id in song_ids]
        
    def get_audio_segment(self, audio_path:Path, beat_time:np.array, start_idx:int, end_idx:int) -> np.ndarray:
        start_time, end_time = beat_time[start_idx], beat_time[end_idx]
        audio, sample_rate = librosa.load(str(audio_path),
                                          sr=self.config.dataset.sample_rate,
                                          offset=start_time,
                                          duration=end_time - start_time)
        return audio
    
    def get_notes_segment(self, notes:np.ndarray, start_idx:int, end_idx:int) -> np.ndarray:
        notes_segment = [note for note in notes if 
                                  any([start_idx <= note[0] < end_idx, start_idx < note[1] <= end_idx])]
        # fix onsets
        for note in notes_segment:
            note[0] = max(note[0], start_idx)
        return np.array(notes_segment)
        
    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        # assume 1 bar == 4 beats
        n_beats = self.config.dataset.n_bars * 4
        # randomly select n_beats from audio
        beat_time = self.beat_times[index]
        
        while True:
            start_idx = np.random.randint(0, len(beat_time) - n_beats - 1)
            end_idx = start_idx + n_beats
            # select midi notes segment: note onset within segment or note offset within segment
            notes_segment = self.get_notes_segment(self.midi_notes[index], start_idx, end_idx)
            if len(notes_segment) > 0:
                break
        audio = self.get_audio_segment(self.audio_paths[index], beat_time, start_idx, end_idx)
        # x.shape = (length, feat_dim)
        x = self.spectrogram(torch.from_numpy(audio)).transpose(-1, -2)   
        y = torch.from_numpy(self.tokenizer.notes_to_tokens(notes_segment))
        
        return x, y
    
    def __len__(self):
        return len(self.audio_paths)
        

class PopDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, config_path:str="config.yaml"):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        self.train_set, self.val_set = random_split(PopDataset(data_dir, config_path), (0.9, 0.1)) 
        
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            collate_fn=collate_fn)
    

def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    # padded shape: (batch_size, length, feat_dim)
    x_batch = pad_sequence(x_batch, batch_first=True, padding_value=PAD)
    # padded shape: (batch_size, length)
    # All labels set to `-100` are ignored (masked)
    y_batch = pad_sequence(y_batch, batch_first=True, padding_value=-100)
    return x_batch, y_batch