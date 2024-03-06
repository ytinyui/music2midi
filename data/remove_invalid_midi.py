import argparse
import multiprocessing
import warnings
from pathlib import Path

from joblib import Parallel, delayed
from pretty_midi import PrettyMIDI
from tqdm import tqdm

warnings.filterwarnings("ignore")


def main(data_dir: Path, meta_path: Path):
    score_id = meta_path.stem
    midi_path = data_dir / "midi" / f"{score_id}.mid"
    try:
        midi_data = PrettyMIDI(str(midi_path))
        midi_notes = [note for track in midi_data.instruments for note in track.notes]
        if len(midi_notes) == 0:
            raise Exception("No notes in MIDI file")
    except ValueError:
        print(f"Largest tick in MIDI file {midi_path} exceeds MAX_TICK")
    except ZeroDivisionError:
        print(f"Invalid tempo in MIDI file: {midi_path}")
    except Exception:
        print(f"No notes in MIDI file: {midi_path}")
    else:
        return
    midi_path.unlink()
    meta_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    args = parser.parse_args()
    Parallel(n_jobs=multiprocessing.cpu_count() // 2, backend="multiprocessing")(
        delayed(main)(Path(args.data_dir), meta_path)
        for meta_path in tqdm(list(Path(args.data_dir).glob("metadata/*.yaml")))
    )
