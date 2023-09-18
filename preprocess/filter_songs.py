from pathlib import Path
from omegaconf import OmegaConf
import os
import shutil
import argparse
from joblib import Parallel, delayed


def filter_condition(yaml, condition):
    if condition == "melody_chroma_accuracy":
        melody_chroma_accuracy = yaml["eval"]["melody_chroma_accuracy"]
        return melody_chroma_accuracy < 0.15

    if condition == "duration":
        song_dur = yaml["song"]["duration"]
        piano_dur = yaml["piano"]["duration"]
        dur_ratio = abs(song_dur - piano_dur) / song_dur
        return dur_ratio > 0.2

    raise ValueError(f"Unknown condition: {condition}")


def run(file):
    yaml = OmegaConf.load(file)
    if filter_condition(yaml):
        os.remove(file)
        os.remove(file.with_suffix(".wav"))
        shutil.rmtree(file.parent / file.stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None, help="provided audio")
    parser.add_argument(
        "--condition",
        type=str,
        default="melody_chroma_accuracy",
        help="condition to filter, should be one of [melody_chroma_accuracy, duration]",
    )
    args = parser.parse_args()

    file_paths = Path(args.path).glob("*.yaml")
    Parallel(n_jobs=4)(delayed(run)(file, args.condition) for file in file_paths)
