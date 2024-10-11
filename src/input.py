from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio


class ModelInputs(NamedTuple):
    input_waveform: torch.Tensor
    notes_batch: Optional[tuple[np.ndarray]] = None
    cond_index: Optional[torch.Tensor] = None


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
        return : melspec(batch, freq, frame)
        """
        with torch.no_grad():
            x = self.melspectrogram(x.float()).transpose(-2, -1)
            x = x.clamp(min=1e-6).log()
        return x


class Conditioning(nn.Module):
    def __init__(self, n_dim: int, num_embeds: list[int]):
        super().__init__()
        # num_embeds: list of dict size of each embed type
        self.embeds = nn.ModuleList([nn.Embedding(num, n_dim) for num in num_embeds])

    def forward(self, feature: torch.Tensor, indices: torch.Tensor):
        """
        Concatenate embeddings to feature

        feature: (batch, L, n_dim)
        index : (batch, n_index)
        """
        concat_embeds = [embed(indices[:, i]) for i, embed in enumerate(self.embeds)]
        concat_embeds = torch.stack(concat_embeds, dim=1)
        return torch.cat([concat_embeds, feature], dim=1)
