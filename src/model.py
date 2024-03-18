import io
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
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from transformers.optimization import Adafactor, AdafactorSchedule

from data.align_audio_midi import get_warp_path

from .evaluation import evaluate_batch
from .input import ModelInputs
from .transformer import T5Transformer, VisionTransformer
from .utils import numpy_to_midi, to_stereo


class Music2Midi(pl.LightningModule):
    def __init__(self, config_path: str):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        model_class = eval(self.config.model.class_)
        assert model_class in [T5Transformer, VisionTransformer]
        self.model: Union[T5Transformer, VisionTransformer] = model_class(config_path)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), warmup_init=True)
        scheduler = AdafactorSchedule(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, inputs: ModelInputs, batch_idx):
        outputs = self.model(inputs)
        loss = outputs.loss
        self.log("train/ce_loss", loss, prog_bar=True)

        if "ct_loss" in outputs.keys():
            ct_loss = outputs.ct_loss
            self.log("train/ct_loss", ct_loss)
            loss = loss + ct_loss

        if (self.global_step + 1) % self.config.trainer.log_every_n_steps == 0:
            metrics, _, _ = self.evaluate_batch(inputs)
            for k, v in metrics.items():
                self.log(f"train/melody_{k}", v)
        return loss

    def validation_step(self, inputs: ModelInputs, batch_idx):
        outputs = self.model(inputs)
        loss = outputs.loss
        self.log("val/ce_loss", loss, prog_bar=True)

        if "ct_loss" in outputs.keys():
            ct_loss = outputs.ct_loss
            self.log("val/ct_loss", ct_loss)
            loss = loss + ct_loss

        metrics, _, _ = self.evaluate_batch(inputs)
        for k, v in metrics.items():
            self.log(f"val/melody_{k}", v)
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
