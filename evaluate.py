import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from music2midi.evaluation import evaluate_batch
from music2midi.model import Music2MIDI
from music2midi.utils import numpy_to_midi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--name", type=str, default="music2midi")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split = np.load(data_dir / "dataset_split.npz", allow_pickle=True)
    test_ids = split.get("test_id")
    model_name = args.name

    config = OmegaConf.load(args.config)
    model = Music2MIDI.load_from_checkpoint(args.ckpt, config_path=args.config).cuda()
    model.eval()

    logs = []
    p_bar = tqdm(test_ids)
    for piano_id in p_bar:
        meta = OmegaConf.load(data_dir / "metadata" / (piano_id + ".yaml"))
        genre = meta.piano.genre
        difficulty = meta.piano.difficulty
        cond_index = [
            config.conditioning.genre.index(genre),
            config.conditioning.difficulty.index(difficulty),
        ]
        label_midi = np.load(data_dir / "midi_numpy" / f"{piano_id}.npy")
        label_midi = numpy_to_midi(label_midi)
        audio_path = data_dir / "audio" / f"{piano_id}.wav"
        output_midi = model.generate(audio_path=audio_path, cond_index=cond_index)

        score = evaluate_batch([label_midi], [output_midi])
        logs.append([piano_id, model_name, genre, difficulty, score])
        p_bar.set_description(f"sample id: {piano_id}, score: {score:.4f}")

    df = pd.DataFrame(
        logs, columns=["piano_id", "model", "genre", "difficulty", "score"]
    )
    df.to_csv(f"score-{model_name}.csv", index=False)
