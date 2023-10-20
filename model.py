from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import librosa
import more_itertools
import numpy as np
import pretty_midi
import pytorch_lightning as pl
import soundfile as sf
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Config, T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule

import wandb
from dataset import PAD, InputDataTuple
from layer.input import LogMelSpectrogram, MelConditioner
from midi_tokenizer import TokenizerFactory
from utils.dsp import to_stereo


class TransformerWrapper(pl.LightningModule):
    def __init__(self, config_path: str):
        super().__init__()

        self.config = OmegaConf.load(config_path)
        self.t5config = T5Config.from_pretrained("t5-small")
        self.PAD = self.t5config.pad_token_id
        for k, v in self.config.t5.items():
            self.t5config.__setattr__(k, v)

        self.t5model = T5ForConditionalGeneration(self.t5config)
        self.tokenizer = TokenizerFactory.create_tokenizer(self.config.tokenizer)

        n_genre = len(self.config.genre_id.keys())
        n_difficulty = len(self.config.difficulty_id.keys())
        self.mel_conditioner = MelConditioner(n_genre, n_difficulty)
        self.spectrogram = LogMelSpectrogram(
            sample_rate=self.config.dataset.sample_rate,
            n_mels=self.t5config.d_model,
            **self.config.spectrogram,
        )

    def setup(self, stage=None):
        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_dict = OmegaConf.to_container(self.config)
        wandb.init(project="pop2piano", name=now_time, config=config_dict)

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), warmup_init=True)
        scheduler = AdafactorSchedule(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch: InputDataTuple, batch_idx):
        x = batch.x
        y = batch.y
        genre_id = batch.genre_id
        difficulty_id = batch.difficulty_id
        x = self.spectrogram(x).transpose(-1, -2)
        x = self.mel_conditioner(x, genre_id, difficulty_id)
        y_pred = self.t5model(inputs_embeds=x, labels=y)

        loss = y_pred.loss
        self.log("train_loss", loss, prog_bar=True)
        wandb.log({"epoch": self.current_epoch, "train_loss": loss})

        return loss

    def validation_step(self, batch: InputDataTuple, batch_idx):
        x = batch.x
        y = batch.y
        genre_id = batch.genre_id
        difficulty_id = batch.difficulty_id
        x = self.spectrogram(x).transpose(-1, -2)
        x = self.mel_conditioner(x, genre_id, difficulty_id)
        y_pred = self.t5model(inputs_embeds=x, labels=y)

        loss = y_pred.loss
        self.log("val_loss", loss, prog_bar=True)
        wandb.log({"epoch": self.current_epoch, "val_loss": loss})

        return loss

    def test_step(self, batch: InputDataTuple, batch_idx):
        x = batch.x
        y = batch.y
        genre_id = batch.genre_id
        difficulty_id = batch.difficulty_id
        x = self.spectrogram(x).transpose(-1, -2)
        x = self.mel_conditioner(x, genre_id, difficulty_id)
        y_pred = self.t5model(inputs_embeds=x, labels=y)

        loss = y_pred.loss
        self.log("test_loss", loss, prog_bar=True)
        wandb.log({"epoch": self.current_epoch, "test_loss": loss})

        return loss

    @torch.no_grad()
    def generate(
        self,
        audio_path: Union[str, Path],
        genre_id: int = 0,
        difficulty_id: int = 0,
        midi_path: Optional[Union[str, Path]] = None,
        mix_path: Optional[Union[str, Path]] = None,
        show_plot: bool = False,
    ) -> pretty_midi.PrettyMIDI:
        audio_y, sr = librosa.load(str(audio_path), sr=self.config.dataset.sample_rate)
        waveform = torch.from_numpy(audio_y).to(self.device)
        duration_per_batch = self.config.dataset.segment_duration
        tokens = self.sample_tokens(
            waveform,
            sr,
            duration_per_batch,
            genre_id=genre_id,
            difficulty_id=difficulty_id,
        )
        numpy_notes = self.tokenizer.batch_decode(
            tokens, duration_per_batch=duration_per_batch
        )
        midi_data = notes_to_midi(numpy_notes)

        if show_plot or mix_path is not None:
            pm_y = midi_data.fluidsynth(sr)
            stereo = to_stereo(audio_y, pm_y)

        if show_plot:
            import IPython.display as ipd
            import note_seq
            from IPython.display import display

            display("Stereo MIX", ipd.Audio(stereo, rate=sr))
            display("Rendered MIDI", ipd.Audio(pm_y, rate=sr))
            display("Original Song", ipd.Audio(audio_y, rate=sr))
            display(note_seq.plot_sequence(note_seq.midi_to_note_sequence(midi_data)))

        if mix_path is not None:
            sf.write(
                file=str(mix_path),
                data=stereo.T,
                samplerate=sr,
                format="wav",
            )
        if midi_path is not None:
            midi_data.write(str(midi_path))

        return midi_data

    @torch.no_grad()
    def sample_tokens(
        self,
        waveform: torch.Tensor,
        sr: int,
        duration_per_batch: float,
        genre_id: int = 0,
        difficulty_id: int = 0,
    ) -> np.ndarray:
        input_batch = torch.split(waveform, sr * duration_per_batch)

        tokens_list = []
        for batch in more_itertools.chunked(
            input_batch, self.config.inference.batch_size
        ):
            batch_size = len(batch)
            inputs_embeds = pad_sequence(batch, batch_first=True, padding_value=PAD)
            genre_ids = torch.zeros((batch_size, 1)).long() + genre_id
            difficulty_ids = torch.zeros((batch_size, 1)).long() + difficulty_id

            inputs_embeds = inputs_embeds.to(self.device)
            genre_ids = genre_ids.to(self.device)
            difficulty_ids = difficulty_ids.to(self.device)

            inputs_embeds = self.spectrogram(inputs_embeds).transpose(-1, -2)
            inputs_embeds = self.mel_conditioner(
                inputs_embeds, genre_ids, difficulty_ids
            )
            tokens = self.t5model.generate(
                inputs_embeds=inputs_embeds,
                max_length=256,
            )
            tokens_list += [*tokens.cpu().numpy()]

        return tokens_list


def notes_to_midi(notes: np.ndarray):
    midi_data = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
    new_inst = pretty_midi.Instrument(program=0, name="Piano")

    new_inst.notes = [
        pretty_midi.Note(
            start=onset_time,
            end=offset_time,
            pitch=int(pitch),
            velocity=int(velocity),
        )
        for onset_time, offset_time, pitch, velocity in notes
    ]
    midi_data.instruments.append(new_inst)
    midi_data.remove_invalid_notes()
    return midi_data
