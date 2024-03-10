from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio


class ModelInputs(NamedTuple):
    input_waveform: torch.Tensor
    notes_batch: Optional[tuple[np.ndarray]] = None
    notes_waveform: Optional[torch.Tensor] = None
    genre_id: Optional[torch.LongTensor] = None
    difficulty_id: Optional[torch.LongTensor] = None


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        f_min: float,
        n_mels: int,
    ):
        super().__init__()
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            n_mels=n_mels,
        )

    def forward(self, x):
        """
        x : waveform(batch, sample)
        return : melspec(batch, 1, freq, frame)
        """
        with torch.no_grad():
            x = self.melspectrogram(x.float())
            x = x.clamp(min=1e-6).log()

        return x.unsqueeze(1)


class MelConditioner(nn.Module):
    def __init__(self, n_genre: int, n_difficulty: int, n_dim: int):
        super().__init__()
        self.embedding_genre = nn.Embedding(num_embeddings=n_genre, embedding_dim=n_dim)
        self.embedding_difficulty = nn.Embedding(
            num_embeddings=n_difficulty, embedding_dim=n_dim
        )

    def forward(
        self,
        feature: torch.Tensor,
        genre_index: torch.Tensor,
        difficulty_index: torch.Tensor,
    ):
        """
        Concatenate embeddings to feature

        feature: (batch, L, n_dim)
        index : (batch, )
        """
        # (batch, 1, feature_dim)
        embedding_genre = self.embedding_genre(genre_index)
        embedding_difficulty = self.embedding_difficulty(difficulty_index)
        return torch.cat([embedding_genre, embedding_difficulty, feature], dim=1)
