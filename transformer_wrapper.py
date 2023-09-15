import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration, Adafactor
from midi_tokenizer import MidiTokenizer, extrapolate_beat_times
from omegaconf import OmegaConf
from preprocess.beat_quantizer import extract_rhythm, interpolate_beat_times
from torch.nn.utils.rnn import pad_sequence
from layer.input import LogMelSpectrogram
import torch
import librosa
import numpy as np
from utils.dsp import get_stereo
import soundfile as sf
from pathlib import Path


class TransformerWrapper(pl.LightningModule):
    def __init__(self, config_path:str):
        super().__init__()
        self.config = OmegaConf.load(config_path)
        self.t5config = T5Config.from_pretrained("t5-small")
        self.PAD = self.t5config.pad_token_id
        for k, v in self.config.t5.items():
            self.t5config.__setattr__(k, v)
            
        self.tokenizer = MidiTokenizer(self.config.tokenizer)
        self.spectrogram = LogMelSpectrogram()
        self.transformer = T5ForConditionalGeneration(self.t5config)
        
    def configure_optimizers(self):
        return Adafactor(self.parameters())
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.transformer(inputs_embeds=x, labels=y)
        loss = y_pred.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.transformer(inputs_embeds=x, labels=y)
        loss = y_pred.loss
        self.log("valid_loss", loss, prog_bar=True)
        return loss
    
    @torch.no_grad()
    def debug(self, idx=1):
        from dataset import PopDataset
        
        data_set = PopDataset("../pop2piano-dataset", "config.yaml")
        x, y = data_set[idx]
        relative_tokens = self.transformer.generate(inputs_embeds=x.unsqueeze(0).to(self.device)).cpu()
        print(relative_tokens)
        print(y)
        notes_hat = self.tokenizer.relative_tokens_to_notes(relative_tokens[0].numpy(), start_idx=0)
        notes = self.tokenizer.relative_tokens_to_notes(y.cpu().numpy(), start_idx=0)
        print(notes_hat)
        print(notes)
    
    @torch.no_grad()
    def generate(
        self,
        audio_path:str,
        audio_sr:int=None,
        model_name="generated",
        stereo_amp=0.5,
        show_plot=False,
        save_midi=False,
        save_mix=False,
        midi_path=None,
        mix_path=None,
        click_amp=0.2,
        add_click=False,
        beatsteps=None,
        mix_sample_rate=None,
    ):
        sr = self.config.dataset.sample_rate
        mix_sample_rate = (sr if mix_sample_rate is None else mix_sample_rate)
        
        audio = get_audio(audio_path, config_sr=sr, audio_sr=audio_sr)
        audio, beatsteps = get_beatsteps(y=audio, audio_path=audio_path, sr=sr)
        audio = torch.from_numpy(audio).to(self.device)
        
        relative_tokens, notes, pm = self.single_inference(
            audio=audio,
            beatsteps=beatsteps,
            midi_path=midi_path,
        )
        
        for n in pm.instruments[0].notes:
            n.start += beatsteps[0]
            n.end += beatsteps[0]

        if show_plot or save_mix:
            if mix_sample_rate != sr:
                y = librosa.core.resample(y, orig_sr=sr, target_sr=mix_sample_rate)
                sr = mix_sample_rate
            if add_click:
                clicks = (
                    librosa.clicks(times=beatsteps, sr=sr, length=len(y)) * click_amp
                )
                y = y + clicks
            pm_y = pm.fluidsynth(sr)
            stereo = get_stereo(y, pm_y, pop_scale=stereo_amp)

        if show_plot:
            import IPython.display as ipd
            from IPython.display import display
            import note_seq

            display("Stereo MIX", ipd.Audio(stereo, rate=sr))
            display("Rendered MIDI", ipd.Audio(pm_y, rate=sr))
            display("Original Song", ipd.Audio(y, rate=sr))
            display(note_seq.plot_sequence(note_seq.midi_to_note_sequence(pm)))


        mix_path = f"{Path(audio_path).stem}_{model_name}_mix.wav" if mix_path is None else mix_path
        if save_mix:
            sf.write(
                file=mix_path,
                data=stereo.T,
                samplerate=sr,
                format="wav",
            )

        # midi_path = f"{Path(audio_path).stem}_{model_name}.mid" if midi_path is None else midi_path
        midi_path = "generated.mid"
        if save_midi:
            pm.write(midi_path)

        return pm, mix_path, midi_path
        
    @torch.no_grad()
    def single_inference(self, audio:torch.Tensor, beatsteps:np.ndarray,midi_path=None):
        n_bars = self.config.dataset.n_bars
        
        inputs_embeds, ext_beatsteps = self.prepare_inference_mel(
            audio,
            beatsteps - beatsteps[0]
        )
        relative_tokens = self.sample_relative_tokens(inputs_embeds)
        
        pm, notes = self.tokenizer.relative_batch_tokens_to_midi(
            relative_tokens,
            beatstep=ext_beatsteps,
            bars_per_batch=n_bars,
            cutoff_time_idx=(n_bars + 1) * 4,
        )

        return relative_tokens, notes, pm
        
    def prepare_inference_mel(self, audio:torch.Tensor, beatsteps:np.ndarray):
        n_bars = self.config.dataset.n_bars
        ext_beatsteps = extrapolate_beat_times(beatsteps, (n_bars + 1) * 4 + 1)
        
        inputs_batch = pack_audio_as_batch(
            audio=audio,
            beatsteps=ext_beatsteps,
            n_steps=n_bars * 4,
            n_target_step=len(beatsteps),
            sample_rate=self.config.dataset.sample_rate
            )
        inputs_batch = pad_sequence(inputs_batch, batch_first=True, padding_value=self.PAD)
        inputs_embeds = self.spectrogram(inputs_batch).transpose(-1, -2)
        
        return inputs_embeds, ext_beatsteps
    
    def sample_relative_tokens(self, inputs_embeds:torch.Tensor):
        inputs_batch_size = inputs_embeds.shape[0]
        inference_batch_size = self.config.inference.batch_size
        
        _relative_tokens_list = []
        for i in range(0, inputs_batch_size, inference_batch_size):
            start = i
            end = min(inputs_batch_size, i + inference_batch_size)

            relative_tokens = self.transformer.generate(
                inputs_embeds=inputs_embeds[start:end],
                max_length=256,
            )
            _relative_tokens_list.append(relative_tokens.cpu().numpy())
            
        max_length = max([rt.shape[-1] for rt in _relative_tokens_list])
        pad_tokens_fn = lambda x: np.pad(x, [(0, 0), (0, max_length - x.shape[-1])], constant_values=self.PAD)
        relative_tokens_list = list(map(pad_tokens_fn, _relative_tokens_list))
        
        return np.concatenate(relative_tokens_list)
    
    
def get_audio(audio_path:str, config_sr:int, audio_sr:int) -> np.ndarray:
    audio_sr = 44100 if audio_sr is None else audio_sr
    if config_sr == audio_sr:
        y, sr = librosa.load(audio_path, sr=audio_sr)
        return y
    
    y, sr = librosa.load(audio_path, sr=audio_sr)
    audio = librosa.core.resample(y, orig_sr=sr, target_sr=config_sr)
    return audio

def get_beatsteps(y:np.ndarray, audio_path:str, sr:int) -> np.ndarray:
    (
        bpm,
        beat_times,
        confidence,
        estimates,
        essentia_beat_intervals,
    ) = extract_rhythm(audio_path, y=y)
    beatsteps = interpolate_beat_times(beat_times, steps_per_beat=2, extend=True)
    
    start_sample = int(beatsteps[0] * sr)
    end_sample = int(beatsteps[-1] * sr)
    
    return y[start_sample:end_sample], beatsteps

def pack_audio_as_batch(
    audio:torch.Tensor,
    beatsteps:np.ndarray,
    n_steps:int,
    n_target_step:int,
    sample_rate:int,
    ) -> list:
    batch = []
    for i in range(0, n_target_step, n_steps):
        start_idx = i
        end_idx = min(i + n_steps, n_target_step)

        start_sample = int(beatsteps[start_idx] * sample_rate)
        end_sample = int(beatsteps[end_idx] * sample_rate)
        feature = audio[start_sample:end_sample]
        batch.append(feature)
        
    return batch


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path('lightning_logs/version_0/checkpoints').glob("*.ckpt").__next__()
    
    wrapper = TransformerWrapper(config_path="config.yaml")
    wrapper = wrapper.load_from_checkpoint(str(ckpt_path), config_path="config.yaml").to(device)
    wrapper.eval()
    print("model loaded")
    
    wrapper.debug()