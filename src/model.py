import io
import multiprocessing
from contextlib import redirect_stdout
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

        encoder_hidden_size = self.encoder_config.hidden_size
        decoder_hidden_size = self.decoder_config.n_embd
        self.encoder = Swinv2Model(self.encoder_config)
        self.proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)
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
            n_dim=decoder_hidden_size,
        )

    def forward(self, inputs: ModelInputs, **kwargs) -> dict:
        labels = self.tokenizer(inputs.notes_batch, padding="left")
        max_length = self.decoder.config.n_positions
        if labels.shape[-1] > max_length:
            labels = labels[:, :max_length]
        if inputs.input_waveform is None:
            return self.decoder(input_ids=labels, labels=labels, **kwargs)

        encoder_inputs = self.spectrogram(inputs.input_waveform)
        b, _, h, w = encoder_inputs.shape
        assert (
            h == w == self.encoder_config.image_size
        ), f"(n_mels, sequence length) = {(h,w)} and encoder image size ({self.encoder_config.image_siz}) mismatch"

        encoder_out = self.encoder(encoder_inputs)
        hidden_states = self.proj(encoder_out.last_hidden_state)
        if inputs.genre_id is None or inputs.difficulty_id is None:
            hidden_states_conditioned = hidden_states
        else:
            hidden_states_conditioned = self.mel_conditioner(
                hidden_states, inputs.genre_id, inputs.difficulty_id
            )

        labels = labels.to(self.decoder.device)
        attention_mask = labels != self.decoder.config.pad_token_id

        decoder_outputs = self.decoder(
            input_ids=labels,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=hidden_states_conditioned,
            **kwargs,
        )

        if inputs.notes_waveform is not None:
            notes_spectrogram = self.spectrogram(inputs.notes_waveform)
            encoder_out2 = self.encoder(notes_spectrogram)
            ct_loss = self.contrastive_loss(
                encoder_out.last_hidden_state, encoder_out2.last_hidden_state
            )
            decoder_outputs["ct_loss"] = ct_loss

        return decoder_outputs

    def generate(self, inputs: ModelInputs, **kwargs) -> torch.Tensor:
        encoder_inputs = self.spectrogram(inputs.input_waveform)
        b, _, h, w = encoder_inputs.shape
        assert (
            h == w == self.encoder_config.image_size
        ), f"(n_mels, sequence length) = {(h,w)} and encoder image size ({self.encoder_config.image_siz}) mismatch"

        encoder_out = self.encoder(encoder_inputs)
        hidden_states = self.proj(encoder_out.last_hidden_state)
        if inputs.genre_id is None and inputs.difficulty_id is None:
            hidden_states_conditioned = hidden_states
        else:
            hidden_states_conditioned = self.mel_conditioner(
                hidden_states, inputs.genre_id, inputs.difficulty_id
            )

        decoder_outputs = self.decoder.generate(
            input_ids=torch.ones((b, 1)).long().to(self.decoder.device),
            encoder_hidden_states=hidden_states_conditioned,
            **kwargs,
        )
        return decoder_outputs


class Music2Midi(pl.LightningModule):
    def __init__(self, config_path: str):
        super().__init__()
        self.config = OmegaConf.load(config_path)
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
        self.log("ce_loss_train", loss, prog_bar=True)
        wandb_logs["ce_loss_train"] = loss

        if "ct_loss" in outputs.keys():
            ct_loss = outputs.ct_loss
            self.log("ct_loss_train", ct_loss, prog_bar=True)
            wandb_logs["ct_loss_train"] = ct_loss

        wandb.log(wandb_logs)
        return loss + ct_loss

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

        wandb.log(wandb_logs)
        return loss + ct_loss

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
        split_size = int(sr * self.config.dataset.segment_duration)
        pad_length = np.ceil(len(audio_y) / split_size).astype(int) * split_size - len(
            audio_y
        )
        audio_y = np.pad(audio_y, (0, pad_length), "constant")
        waveform = torch.from_numpy(audio_y).to(self.device)
        split_duration = self.config.dataset.segment_duration
        numpy_notes = self.sample_tokens(
            waveform,
            split_size,
            split_duration=split_duration,
            genre_id=genre_id,
            difficulty_id=difficulty_id,
        )
        # if self.model.tokenizer.time_step:
        #     numpy_notes = self.model.tokenizer.decode(
        #         tokens_list, mode="sequential", split_duration=split_duration
        #     )
        # else:
        #     numpy_notes_split = Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
        #         delayed(self.output_dtw)(waveform.cpu().numpy(), tokens, sr)
        #         for waveform, tokens in zip(input_split, tokens_list)
        #     )
        #     # add time offset
        #     for i, _ in enumerate(numpy_notes_split):
        #         numpy_notes_split[i][:, :2] += i * split_duration
        #     numpy_notes = np.concatenate(numpy_notes_split)

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
        split_size: float,
        split_duration: float,
        genre_id: int = 0,
        difficulty_id: int = 0,
    ) -> np.array:
        """
        Return: input tuple[waveform split, list of tokens]
        """
        input_split = torch.split(waveform, split_size)

        tokens_list = []
        for batch in more_itertools.chunked(
            input_split, self.config.inference.batch_size
        ):
            batch_size = len(batch)
            input_wav = pad_sequence(batch, batch_first=True, padding_value=0)
            genre_ids = torch.zeros((batch_size, 1)).long() + genre_id
            difficulty_ids = torch.zeros((batch_size, 1)).long() + difficulty_id

            input_wav = input_wav.to(self.device)
            genre_ids = genre_ids.to(self.device)
            difficulty_ids = difficulty_ids.to(self.device)

            model_inputs = ModelInputs(
                input_waveform=input_wav,
                genre_id=genre_ids,
                difficulty_id=difficulty_ids,
            )
            tokens = self.model.generate(
                model_inputs,
                max_length=1024,
            )
            tokens_list += [*tokens]

        numpy_notes = self.model.tokenizer.decode(
            tokens_list, mode="sequential", duration_per_batch=split_duration
        )
        return numpy_notes

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
