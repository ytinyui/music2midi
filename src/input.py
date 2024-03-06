import torch
import torch.nn as nn
import torchaudio


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
        # x : audio(batch, sample)
        # X : melspec (batch, freq, frame)
        with torch.no_grad():
            X = self.melspectrogram(x)
            X = X.clamp(min=1e-6).log()

        return X


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
