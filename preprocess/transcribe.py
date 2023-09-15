from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import argparse
from pathlib import Path
from contextlib import redirect_stdout
import io
import warnings
from tqdm import tqdm


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None, help="provided audio")
    args = parser.parse_args()

    path = Path(args.path)
    files = path.glob("*.wav")

    transcriptor = PianoTranscription(device="cuda", checkpoint_path=None)
    for file in tqdm(list(files)):
        midi_file = (path / file).with_suffix(".mid")
        audio, _ = load_audio(str(path / file), sr=sample_rate, mono=True)
        with redirect_stdout(io.StringIO()):
            transcribed_dict = transcriptor.transcribe(audio, str(midi_file))
