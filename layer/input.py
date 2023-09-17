import torch
import torch.nn as nn
import torchaudio


class LogMelSpectrogram(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=4096,
            hop_length=1024,
            f_min=10.0,
            n_mels=512,
        )

    def forward(self, x):
        # x : audio(batch, sample)
        # X : melspec (batch, freq, frame)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                X = self.melspectrogram(x)
                X = X.clamp(min=1e-6).log()

        return X


class MelConditioner(nn.Module):
    def __init__(self, n_vocab, n_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_dim)

    def forward(self, feature: torch.Tensor, index: torch.Tensor):
        """
        Concatenate composer embedding to feature

        feature: (batch, L, n_dim)
        index : (batch, )
        """
        # (batch, 1, feature_dim)
        composer_embedding = self.embedding(index)
        return torch.cat([composer_embedding, feature], dim=1)
