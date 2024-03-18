import argparse
import multiprocessing
import os
import subprocess
from pathlib import Path

import soundfile as sf
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from pretty_midi import PrettyMIDI
from tqdm import tqdm


def main(
    song_path: Path,
    data_dir: Path,
    sr: int = 22050,
    output_sr: int = 22050,
    output_mono: bool = True,
):
    piano_id = song_path.stem
    midi_path = data_dir / "midi_aligned" / f"{piano_id}.mid"
    if not midi_path.exists():
        print(f"{midi_path} file not found")
        return
    mix_path = data_dir / "audio_mix" / f"{piano_id}.mp3"
    midi_data = PrettyMIDI(str(midi_path))
    midi_synth = midi_data.synthesize(fs=sr)

    midi_synth_path = mix_path.with_name(piano_id + "_midi.wav")
    sf.write(str(midi_synth_path), midi_synth, sr)

    filter_complex = (
        "[0:a]volume=-5dB[a0];[1:a]volume=15dB[a1],[a0][a1]amix=inputs=2:duration=longest,loudnorm"
        if output_mono
        else "amerge=inputs=2,pan=stereo|c0<0.5*c0|c1<1.5*c1,loudnorm"
    )
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-i",
            str(song_path),
            "-i",
            str(midi_synth_path),
            "-filter_complex",
            filter_complex,
            "-ar",
            str(output_sr),
            str(mix_path),
            "-loglevel",
            "error",
            "-y",
        ]
    )
    if process.wait() != 0:
        print(f"{piano_id}: error in ffmpeg")
    os.remove(midi_synth_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    parser.add_argument(
        "--output_sr", type=int, default=None, help="sample rate of mixed audio"
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="if specified, the synthesized MIDI audio and the song audio are mixed into mono audio;\
              otherwise, the output is stereo",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    os.makedirs(data_dir / "audio_mix", exist_ok=True)
    config = OmegaConf.load("config.yaml")
    sr = config.dataset.sample_rate
    output_sr = sr if args.output_sr is None else args.output_sr

    Parallel(n_jobs=multiprocessing.cpu_count(), backend="multiprocessing")(
        delayed(main)(
            song_path, data_dir, sr=sr, output_sr=output_sr, output_mono=args.mono
        )
        for song_path in tqdm(list(data_dir.glob("audio/*.wav")))
    )
