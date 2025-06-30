import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config = OmegaConf.load("config.yaml")
    threshold = config.dataset.filter_threshold
    meta_list = [
        meta
        for meta_path in data_dir.glob("metadata/*.yaml")
        if (meta := OmegaConf.load(meta_path)).get("youtube") is not None
        and meta.metrics.opt_chroma_shift == 0
        and meta.piano.num_tracks == 2
    ]
    metrics = [*meta_list[0].metrics.keys()]
    df = pd.DataFrame(
        [[meta.piano.id] + [*meta.metrics.values()] for meta in meta_list],
        columns=["piano_id"] + metrics,
    )
    df_filtered = df[
        (df["wp_std"] < threshold["wp_std"])
        & (df["max_beat_fluctuation"] < threshold["max_beat_fluctuation"])
        & (df["max_note_density"] < threshold["max_note_density"])
        & (df["time_diff_ratio"] < threshold["time_diff_ratio"])
    ]

    dataset_ids = df_filtered["piano_id"].to_numpy()
    train_ids, test_ids = train_test_split(dataset_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

    np.savez(
        data_dir / "dataset_split.npz",
        train_id=train_ids,
        val_id=val_ids,
        test_id=test_ids,
    )
