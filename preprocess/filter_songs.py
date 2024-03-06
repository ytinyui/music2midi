from pathlib import Path
from omegaconf import OmegaConf
import os
import shutil
import argparse
from joblib import Parallel, delayed


def filter_condition(yaml):
    melody_chroma_accuracy = yaml["eval"]["melody_chroma_accuracy"]
    song_dur = yaml["song"]["duration"]
    piano_dur = yaml["piano"]["duration"]
    dur_ratio = abs(song_dur - piano_dur) / song_dur
    return melody_chroma_accuracy < 0.15 or dur_ratio > 0.2


def run(file):
    yaml = OmegaConf.load(file)
    if filter_condition(yaml):
        os.remove(file)
        os.remove(file.with_suffix(".wav"))
        shutil.rmtree(file.parent / file.stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None, help="provided audio")
    args = parser.parse_args()

    file_paths = Path(args.path).glob("*.yaml")
    Parallel(n_jobs=4)(delayed(run)(file) for file in file_paths)
