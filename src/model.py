import io
import multiprocessing
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional, Union

import librosa
import more_itertools
import numpy as np
import pretty_midi
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Config, GPT2LMHeadModel, Swinv2Config, Swinv2Model
from transformers.optimization import Adafactor, AdafactorSchedule

import wandb
from data.align_audio_midi import get_warp_path

from .input import LogMelSpectrogram, MelConditioner, ModelInputs
from .tokenizer import MidiTokenizer
from .utils import numpy_to_midi, to_stereo


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        assert x1.shape == x2.shape
        sim = F.cosine_similarity(x1.unsqueeze(0), x2.unsqueeze(1), dim=-1).mean(dim=-1)
        return F.cross_entropy(sim / self.temp, torch.arange(x1.shape[0]).to(x1.device))


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, config_path: str):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        self.encoder_config = Swinv2Config(**self.config.swinv2)
        self.decoder_config = GPT2Config.from_pretrained("gpt2")
        for k, v in self.config.gpt2.items():
            setattr(self.decoder_config, k, v)

        self.encoder = Swinv2Model(self.encoder_config)
        self.decoder = GPT2LMHeadModel(self.decoder_config)

        self.tokenizer = MidiTokenizer(self.config)
        self.spectrogram = LogMelSpectrogram(
            sample_rate=self.config.dataset.sample_rate,
            n_mels=self.config.swinv2.image_size,
            **self.config.spectrogram,
        )
        self.contrastive_loss = ContrastiveLoss()

        n_genre = len(self.config.genre_id.keys())
        n_difficulty = len(self.config.difficulty_id.keys())
        self.mel_conditioner = MelConditioner(
            n_genre,
            n_difficulty,
            n_dim=self.config.gpt2.n_embd,
        )

    def forward(self, inputs: ModelInputs, **kwargs) -> dict:
        if inputs.input_waveform is None:
            return self.decoder(input_ids=labels, labels=labels, **kwargs)

        encoder_inputs = self.spectrogram(inputs.input_waveform)
        b, _, h, w = encoder_inputs.shape
        assert (
            h == w == self.encoder_config.image_size
        ), f"n_mels({h}) does not match sequence length({w})"

        encoder_last_hidden_state = self.encoder(encoder_inputs).last_hidden_state
        if inputs.genre_id is None and inputs.difficulty_id is None:
            encoder_hidden_states = encoder_last_hidden_state
        else:
            encoder_hidden_states = self.mel_conditioner(
                encoder_last_hidden_state, inputs.genre_id, inputs.difficulty_id
            )

        labels = self.tokenizer(inputs.notes_batch, padding="left")
        labels = labels.to(self.decoder.device)
        attention_mask = labels != self.decoder.config.pad_token_id

        decoder_outputs = self.decoder(
            input_ids=labels,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )

        if inputs.notes_waveform is not None:
            notes_spectrogram = self.spectrogram(inputs.notes_waveform)
            ct_loss = self.contrastive_loss(
                encoder_last_hidden_state,
                self.encoder(notes_spectrogram).last_hidden_state,
            )
            decoder_outputs["ct_loss"] = ct_loss

        return decoder_outputs

    def generate(self, inputs: ModelInputs, **kwargs) -> torch.Tensor:
        encoder_inputs = self.spectrogram(inputs.input_waveform)
        b, _, h, w = encoder_inputs.shape
        assert h == w == self.encoder_config.image_size

        encoder_last_hidden_state = self.encoder(encoder_inputs).last_hidden_state
        if inputs.genre_id is None and inputs.difficulty_id is None:
            encoder_hidden_states = encoder_last_hidden_state
        else:
            encoder_hidden_states = self.mel_conditioner(
                encoder_last_hidden_state, inputs.genre_id, inputs.difficulty_id
            )

        decoder_outputs = self.decoder.generate(
            input_ids=torch.ones((b, 1)).long().to(self.decoder.device),
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        return decoder_outputs


class Music2Midi(pl.LightningModule):
    def __init__(self, config_path: str):
        super().__init__()
        self.model = EncoderDecoderTransformer(config_path)

    def setup(self, stage=None):
        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_dict = OmegaConf.to_container(self.model.config)
        wandb.init(project="music2midi", name=now_time, config=config_dict)

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), warmup_init=True)
        scheduler = AdafactorSchedule(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, inputs: ModelInputs, batch_idx):
        outputs = self.model(inputs)
        wandb_logs = {"epoch": self.current_epoch}
        loss = outputs.loss
        self.log("ce_loss", loss, prog_bar=True)
        wandb_logs["ce_loss_train"] = loss

        if "ct_loss" in outputs.keys():
            ct_loss = outputs.ct_loss
            self.log("ct_loss_train", ct_loss, prog_bar=True)
            wandb_logs["ct_loss_train"] = ct_loss
            loss = loss + ct_loss
        wandb.log(wandb_logs)
        return loss

    def validation_step(self, inputs: ModelInputs, batch_idx):
        outputs = self.model(inputs)
        wandb_logs = {"epoch": self.current_epoch}
        loss = outputs.loss
        self.log("ce_loss_val", loss, prog_bar=True)
        wandb_logs["ce_loss_val"] = loss

        if "ct_loss" in outputs.keys():
            ct_loss = outputs.ct_loss
            self.log("ct_loss_val", ct_loss, prog_bar=True)
            wandb_logs["ct_loss_val"] = ct_loss
            loss = loss + ct_loss
        wandb.log(wandb_logs)
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        audio_y: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        genre_id: int = 0,
        difficulty_id: int = 0,
        midi_path: Optional[Union[str, Path]] = None,
        mix_path: Optional[Union[str, Path]] = None,
        show_plot: bool = False,
    ) -> pretty_midi.PrettyMIDI:
        """
        Specify either audio_path or audio_y as input.
        """
        if audio_path is None and audio_y is None:
            raise ValueError("Either audio_path or audio_y should be specified")
        if sr is None:
            sr = self.config.dataset.sample_rate
        if audio_y is None:
            audio_y, sr = librosa.load(str(audio_path), sr=sr)
        waveform = torch.from_numpy(audio_y).to(self.device)
        duration_per_batch = self.config.dataset.segment_duration
        input_split, tokens_list = self.sample_tokens(
            waveform,
            sr,
            duration_per_batch,
            genre_id=genre_id,
            difficulty_id=difficulty_id,
        )
        if self.tokenizer.time_step:
            numpy_notes = self.tokenizer.batch_decode(
                tokens_list, duration_per_batch=duration_per_batch
            )
        else:
            numpy_notes_split = Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
                delayed(self.output_dtw)(waveform.cpu().numpy(), tokens, sr)
                for waveform, tokens in zip(input_split, tokens_list)
            )
            # add time offset
            for i, _ in enumerate(numpy_notes_split):
                numpy_notes_split[i][:, :2] += i * duration_per_batch
            numpy_notes = np.concatenate(numpy_notes_split)

        midi_data = numpy_to_midi(numpy_notes)

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
    ) -> tuple[tuple[torch.Tensor], list[np.ndarray]]:
        """
        Return: input tuple[waveform split, list of tokens]
        """
        input_split = torch.split(waveform, sr * duration_per_batch)

        tokens_list = []
        for batch in more_itertools.chunked(
            input_split, self.config.inference.batch_size
        ):
            batch_size = len(batch)
            inputs_embeds = pad_sequence(batch, batch_first=True, padding_value=0)
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

        return input_split, tokens_list

    def output_dtw(
        self, waveform: np.ndarray, tokens: np.ndarray, sr: int
    ) -> np.ndarray:
        duration = len(waveform) / sr
        numpy_notes = np.float_(self.tokenizer.decode(tokens))
        if len(numpy_notes) == 0:
            return np.zeros((0, 4))
        max_beat_index = max(np.max(numpy_notes[:, :2]), 1)
        numpy_notes[:, :2] *= duration / max_beat_index
        midi_data = numpy_to_midi(numpy_notes)
        midi_synth = midi_data.synthesize(fs=sr)
        with redirect_stdout(io.StringIO()):
            wp, _ = get_warp_path(waveform, midi_synth, sr, strictly_monotonic=True)
        numpy_notes[:, :2] = np.interp(numpy_notes[:, :2], wp[1], wp[0])
        return numpy_notes
