import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from omegaconf import OmegaConf
import yt_dlp
import argparse
import os
from typing import Callable


def get_ydl_opts(
    output_dir: Path,
    output_file_template: str = "%(id)s.%(ext)s",
    process_fn: Callable = lambda x: None,
) -> dict:
    return {
        "final_ext": "wav",
        "format": "bestaudio/best",
        "noprogress": True,
        "outtmpl": {"default": str(output_dir / output_file_template)},
        "postprocessor_args": {"ffmpeg": ["-ac", "1", "-ar", "16000"]},
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "nopostoverwrites": False,
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "match_filter": process_fn,
        "quiet": True,
        "retries": 25,
    }


def record_error(ytid, msg) -> None:
    with open("error.csv", "a") as f:
        f.write(f"{ytid},{msg}\n")
    return


def download_piano(piano_id: str, output_dir: Path) -> None:
    output_dir = output_dir / piano_id
    os.makedirs(output_dir, exist_ok=True)
    piano_file = output_dir / f"{piano_id}.wav"
    if (piano_file).exists():
        print(piano_id, "file already exists (song)")
        return

    def piano_process_fn(info: dict) -> None:
        ytid, uploader, title, duration = (
            info["id"],
            info["uploader"],
            info["title"],
            info["duration"],
        )
        meta = OmegaConf.create()
        meta.piano = OmegaConf.create()
        meta.piano.ytid = ytid
        meta.piano.uploader = uploader
        meta.piano.title = title
        meta.piano.duration = int(duration)
        OmegaConf.save(meta, str((output_dir / piano_id).with_suffix(".yaml")))
        return

    yt_opts = get_ydl_opts(output_dir, process_fn=piano_process_fn)
    with yt_dlp.YoutubeDL(yt_opts) as ydl:
        ydl.download(f"https://www.youtube.com/watch?v={piano_id}")

    if len(list(output_dir.glob(f"{piano_id}.wav"))) == 0:
        record_error(piano_id, "file not found (piano)")
    return


def download_song(piano_id: str, pop_id: str, output_dir: Path) -> None:
    output_dir = output_dir / piano_id
    song_file = output_dir / f"{pop_id}.wav"
    yaml_file = output_dir / f"{piano_id}.yaml"
    if (song_file).exists():
        print(pop_id, "file already exists (song)")
        return
    if (yaml_file).exists() == False:
        print(piano_id, "yaml not found")
        return

    def song_process_fn(info: dict) -> None:
        ytid, uploader, title, duration = (
            info["id"],
            info["uploader"],
            info["title"],
            info["duration"],
        )
        meta = OmegaConf.load(str(yaml_file))
        meta.song = OmegaConf.create()
        meta.song.ytid = ytid
        meta.song.uploader = uploader
        meta.song.title = title
        meta.song.duration = int(duration)
        OmegaConf.save(meta, str(yaml_file))
        return

    yt_opts = get_ydl_opts(output_dir, process_fn=song_process_fn)
    with yt_dlp.YoutubeDL(yt_opts) as ydl:
        ydl.download(f"https://www.youtube.com/watch?v={pop_id}")

    if len(list(output_dir.glob(f"{pop_id}.wav"))) == 0:
        record_error(pop_id, "file not found (song)")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, default=None, help="directory of csv")
    parser.add_argument("output_dir", type=str, default=None, help="output dir")
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=None,
        help="if specified, only {num_pairs} audio pairs will be downloaded",
    )
    parser.add_argument(
        "--parallel",
        default=4,
        type=int,
        help="number of parallel processes (default 4)",
    )
    parser.add_argument(
        "--dur_diff_limit",
        default=20,
        type=int,
        help="upper limit of duration difference between song and piano cover in seconds (default 20)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = Path(args.dataset).glob("*.csv")
    df_list = [pd.read_csv(f) for f in files]

    df_merged = pd.concat(df_list)
    df_merged["dur_diff"] = abs(df_merged["piano_duration"] - df_merged["pop_duration"])
    df_merged = df_merged[df_merged["dur_diff"] <= args.dur_diff_limit]
    df_merged = df_merged[: args.num_pairs]

    print("Downloading piano audio:")
    Parallel(n_jobs=args.parallel)(
        delayed(download_piano)(piano_id, output_dir=Path(args.output_dir))
        for piano_id in tqdm(list(df_merged["piano_ids"]))
    )
    print("Downloading song audio:")
    Parallel(n_jobs=args.parallel)(
        delayed(download_song)(piano_id, pop_id, output_dir=Path(args.output_dir))
        for piano_id, pop_id in tqdm(
            list(zip(df_merged["piano_ids"], df_merged["pop_ids"]))
        )
    )
