import argparse
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from omegaconf import OmegaConf
from pretty_midi import PrettyMIDI
from tqdm import tqdm

from src.model import Music2Midi, numpy_to_midi


def get_highest_pitches_from_piano_roll(piano_roll: np.ndarray) -> np.ndarray:
    """
    return the pitch class (0, 1, ..., 11) of the highest pitch onset at each frame.
    if there is no onset at this frame, return 12.

    input:
        piano_roll : (128, num_frames)

    return: (num_frames, )
    """
    num_frames = piano_roll.shape[1]
    ret = np.ones(num_frames, dtype=np.int_) * 12
    for i in range(num_frames):
        if np.sum(piano_roll[:, i]) == 0:  # no onset at this frame
            continue
        (onset_pitches,) = np.nonzero(piano_roll[:, i])
        ret[i] = onset_pitches[-1] % 12

    return ret


def evaluate(
    label: PrettyMIDI,
    output: PrettyMIDI,
    metric_fn: Callable = partial(metrics.f1_score, average="micro"),
    piano_roll_frame_rate: int = 100,
) -> float:
    end_time = max(output.get_end_time(), label.get_end_time())
    times = np.arange(0, end_time, 1 / piano_roll_frame_rate)

    output_piano_roll = output.get_piano_roll(fs=piano_roll_frame_rate, times=times)
    label_piano_roll = label.get_piano_roll(fs=piano_roll_frame_rate, times=times)

    output_melody = get_highest_pitches_from_piano_roll(output_piano_roll)
    label_melody = get_highest_pitches_from_piano_roll(label_piano_roll)

    return metric_fn(label_melody, output_melody)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--name", type=str, default="M2M")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split = np.load(data_dir / "dataset_split.npz", allow_pickle=True)
    test_ids = split.get("test_id")
    model_name = args.name

    config = OmegaConf.load(args.config)
    model = Music2Midi.load_from_checkpoint(
        args.ckpt_path, config_path=args.config
    ).cuda()
    model.eval()

    logs = []

    for score_id in tqdm(test_ids):
        meta = OmegaConf.load(data_dir / "metadata" / (score_id + ".yaml"))
        genre = meta.score.genre
        difficulty = meta.score.difficulty
        label_midi = np.load(data_dir / "midi_numpy" / f"{score_id}.npy")
        label_midi = numpy_to_midi(label_midi)
        audio_path = data_dir / "audio" / f"{score_id}.wav"

        try:
            output_midi = model.generate(
                audio_path=audio_path,
                genre_id=config.genre_id.get(genre),
                difficulty_id=config.difficulty_id.get(difficulty),
            )
        except Exception as e:
            print(e)
            print(f"Error occurred for {score_id}")
            continue

        result = evaluate(label_midi, output_midi)
        logs.append([score_id, model_name, genre, difficulty, result])

    df = pd.DataFrame(
        logs, columns=["score_id", "model", "genre", "difficulty", "score"]
    )
    df.to_csv(f"score-{model_name}.csv", index=False)
