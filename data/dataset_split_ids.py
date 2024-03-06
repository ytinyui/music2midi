import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


def filter_condition(meta: DictConfig):
    metrics = meta.metrics
    return (
        False
        if any(
            [
                metrics.opt_chroma_shift != 0,
                metrics.chroma_min_similarity < 0.01,
                metrics.tempogram_min_similarity < 0.01,
            ]
        )
        else True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta_list = [
        meta
        for meta_path in data_dir.glob("metadata/*.yaml")
        if (meta := OmegaConf.load(meta_path)).get("youtube") is not None
        and meta.metrics.opt_chroma_shift == 0
    ]
    metrics = [*meta_list[0].metrics.keys()]
    df = pd.DataFrame(
        [[meta.score.id] + [*meta.metrics.values()] for meta in meta_list],
        columns=["score_id"] + metrics,
    )
    df_filtered = df[
        (df["norm_wp_std"] < 0.05)
        & (df["chroma_min_similarity"] > 0.2)
        & (df["tempogram_min_similarity"] > 0.2)
        & (df["note_density"] < 50)
    ]

    dataset_ids = df_filtered["score_id"].to_numpy()
    train_ids, test_ids = train_test_split(dataset_ids, test_size=0.1, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

    np.savez(
        data_dir / "dataset_split.npz",
        train_id=train_ids,
        # val_id=val_ids,
        test_id=test_ids,
    )
