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
        and meta.score.num_tracks == 2
    ]
    metrics = [*meta_list[0].metrics.keys()]
    df = pd.DataFrame(
        [[meta.score.id] + [*meta.metrics.values()] for meta in meta_list],
        columns=["score_id"] + metrics,
    )
    df_filtered = df[
        (df["norm_wp_std"] < threshold["norm_wp_std"])
        & (df["beat_local_fluctuation"] < threshold["beat_local_fluctuation"])
        & (df["piecewise_note_density"] < threshold["piecewise_note_density"])
        & (df["chroma_min_similarity"] > threshold["chroma_min_similarity"])
        & (df["tempogram_min_similarity"] > threshold["tempogram_min_similarity"])
    ]

    dataset_ids = df_filtered["score_id"].to_numpy()
    train_ids, test_ids = train_test_split(dataset_ids, test_size=0.1, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

    np.savez(
        data_dir / "dataset_split.npz",
        train_id=train_ids,
        val_id=val_ids,
        test_id=test_ids,
    )
