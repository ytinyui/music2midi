from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, default=None, help="provided audio")

if __name__ == "__main__":
    args = parser.parse_args()
    path = Path(args.path)
    files = path.glob("*.wav")
    for file in files:
        midi_file = (path / file).with_suffix(".mid")
        if midi_file.exists():
            continue
        # Load audio
        (audio, _) = load_audio(str(path / file), sr=sample_rate, mono=True)

        # Transcriptor
        transcriptor = PianoTranscription(device='cuda', checkpoint_path=None)  # device: 'cuda' | 'cpu'

        # Transcribe and write out to MIDI file
        transcribed_dict = transcriptor.transcribe(audio, str(midi_file))