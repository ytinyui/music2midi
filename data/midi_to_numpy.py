import argparse
import multiprocessing
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from pretty_midi import PrettyMIDI
from tqdm import tqdm


def midi_to_numpy(midi_path: Path):
    midi_data = PrettyMIDI(str(midi_path))
    notes_array = []

    for track in midi_data.instruments:
        for note in track.notes:
            numpy_note = [
                note.start,
                note.end,
                note.pitch,
                note.velocity,
            ]
            notes_array.append(numpy_note)
    notes_array = np.array(notes_array)
    # sort priority: onset time > offset time > pitch
    sorted_indices = np.lexsort([col for col in notes_array.T[2::-1]])
    return notes_array[sorted_indices]


def main(midi_path: Path, output_dir: Path):
    if not midi_path.exists():
        print(f"{midi_path.name} file not found")
        return
    numpy_notes = midi_to_numpy(midi_path)
    np.save(output_dir / (midi_path.stem + ".npy"), numpy_notes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = data_dir / "midi_numpy"
    output_dir.mkdir(exist_ok=True)

    Parallel(n_jobs=multiprocessing.cpu_count() // 2)(
        delayed(main)(midi_path, output_dir)
        for midi_path in tqdm(list(data_dir.glob("midi_aligned/*.mid")))
    )
