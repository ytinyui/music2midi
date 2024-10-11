import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import T5Config, T5ForConditionalGeneration

from .input import Conditioning, LogMelSpectrogram, ModelInputs
from .tokenizer import MidiTokenizer


class T5Transformer(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        self.t5config = T5Config(**self.config.model.t5)

        self.transformer = T5ForConditionalGeneration(self.t5config)
        self.tokenizer = MidiTokenizer(self.config)
        self.spectrogram = LogMelSpectrogram(
            sample_rate=self.config.model.sample_rate,
            n_mels=self.config.model.t5.d_model,
            **self.config.spectrogram,
        )
        self.conditioning = Conditioning(
            self.config.model.t5.d_model,
            [len(v) for v in self.config.conditioning.values()],
        )

    def forward(self, inputs: ModelInputs, **kwargs) -> dict:
        labels = self.tokenizer(inputs.notes_batch)
        labels[labels == self.transformer.config.pad_token_id] = -100
        labels = labels.to(self.transformer.device)

        encoder_inputs = self.spectrogram(inputs.input_waveform)
        encoder_inputs = self.conditioning(encoder_inputs, inputs.cond_index)
        outputs = self.transformer(
            inputs_embeds=encoder_inputs, labels=labels, **kwargs
        )

        return outputs

    def generate(self, inputs: ModelInputs, **kwargs) -> torch.Tensor:
        encoder_inputs = self.spectrogram(inputs.input_waveform)
        encoder_inputs = self.conditioning(encoder_inputs, inputs.cond_index)
        outputs = self.transformer.generate(inputs_embeds=encoder_inputs, **kwargs)
        return outputs
