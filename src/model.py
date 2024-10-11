from pathlib import Path
from typing import Optional, Union

import librosa
import more_itertools
import numpy as np
import pretty_midi
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from transformers.optimization import Adafactor, AdafactorSchedule

from .evaluation import evaluate_batch
from .input import ModelInputs
from .transformer import T5Transformer
from .utils import numpy_to_midi


class Music2MIDI(pl.LightningModule):
    def __init__(self, config_path: str):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        self.model = T5Transformer(config_path)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), warmup_init=True)
        scheduler = AdafactorSchedule(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, inputs: ModelInputs, batch_idx):
        batch_size = inputs.input_waveform.shape[0]
        outputs = self.model(inputs)
        loss = outputs.loss
        self.log(
            "train/loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True
        )

        if (self.global_step + 1) % self.config.trainer.log_every_n_steps == 0:
            score, _, _ = self.evaluate_batch(inputs)
            self.log("train/score", score, batch_size=batch_size, sync_dist=True)
        return loss

    def validation_step(self, inputs: ModelInputs, batch_idx):
        batch_size = inputs.input_waveform.shape[0]
        outputs = self.model(inputs)
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)

        score, _, _ = self.evaluate_batch(inputs)
        self.log("val/score", score, batch_size=batch_size, sync_dist=True)
        return loss

    @torch.no_grad()
    def evaluate_batch(self, inputs: ModelInputs):
        max_num_notes = max(len(notes) for notes in inputs.notes_batch)
        generated_outputs = self.model.generate(inputs, max_length=max_num_notes * 4)
        decoded_notes = self.model.tokenizer.decode(generated_outputs, mode="batched")

        label_midi = [numpy_to_midi(notes) for notes in inputs.notes_batch]
        output_midi = [numpy_to_midi(notes) for notes in decoded_notes]
        metrics = evaluate_batch(label_midi, output_midi)

        return metrics, output_midi, label_midi

    def generate(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        audio_y: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        cond_index: Optional[list[int]] = None,
    ) -> pretty_midi.PrettyMIDI:
        """
        Specify either audio_path or audio_y as input.
        """
        if audio_path is None and audio_y is None:
            raise ValueError("Either audio_path or audio_y should be specified")
        if sr is None:
            sr = self.config.model.sample_rate
        else:
            assert sr == self.config.model.sample_rate
        if audio_y is None:
            audio_y, sr = librosa.load(str(audio_path), sr=sr)
        # Pad the audio to be multiple of split size
        split_size = int(sr * self.config.dataset.segment_duration)
        pad_length = np.ceil(len(audio_y) / split_size).astype(int) * split_size - len(
            audio_y
        )
        audio_y = np.pad(audio_y, (0, pad_length), "constant")
        waveform = torch.from_numpy(audio_y).to(self.device)
        split_duration = self.config.dataset.segment_duration

        numpy_notes = self.sample_tokens(
            waveform, split_size, split_duration=split_duration, cond_index=cond_index
        )
        midi_data = numpy_to_midi(numpy_notes)

        return midi_data

    @torch.no_grad()
    def sample_tokens(
        self,
        waveform: torch.Tensor,
        split_size: int,
        split_duration: float,
        cond_index: Optional[list[int]] = None,
    ) -> np.array:
        """
        Return: input tuple[waveform split, list of tokens]
        """
        n_embeds = len(self.model.conditioning.embeds)
        input_split = torch.split(waveform, split_size)
        tokens_list = []
        for batch in more_itertools.chunked(
            input_split, self.config.inference.batch_size
        ):
            batch_size = len(batch)
            input_wav = pad_sequence(batch, batch_first=True, padding_value=0)

            if cond_index is None:
                cond_index_batch = torch.zeros((batch_size, n_embeds))

            else:
                cond_index_batch = torch.zeros((batch_size, n_embeds)) + torch.Tensor(
                    cond_index
                )
            cond_index_batch = cond_index_batch.long().to(self.device)

            input_wav = input_wav.to(self.device)
            model_inputs = ModelInputs(
                input_waveform=input_wav, cond_index=cond_index_batch
            )
            tokens = self.model.generate(model_inputs, max_length=1024)
            tokens_list += [*tokens]

        numpy_notes = self.model.tokenizer.decode(
            tokens_list, mode="sequential", duration_per_batch=split_duration
        )
        return numpy_notes
