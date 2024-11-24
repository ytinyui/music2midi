import argparse
import functools
import multiprocessing
from pathlib import Path
from typing import Callable, Union

import librosa
import pandas as pd
import yt_dlp
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def get_yt_id(piano_id: str, data_dir: Path) -> str:
    """
    returns the first entry in the csv
    """
    df = pd.read_csv(data_dir / "youtube_csv" / f"{piano_id}.csv")
    if df.empty:
        return ""
    return df.iloc[0].yt_id


def save_metadata(info: dict, meta: DictConfig, meta_path: Path):
    meta.youtube.title = info["title"]
    OmegaConf.save(meta, meta_path)


def get_opts(
    output_file: Union[str, Path],
    save_metadata_fn: Callable,
    sr: int,
    cookie_file: Union[str, None],
    quiet: bool,
) -> dict:
    return {
        "final_ext": "wav",
        "format": "bestaudio/best",
        "noprogress": True,
        "outtmpl": {"default": str(output_file)},
        "postprocessor_args": {"ffmpeg": ["-ac", "1", "-ar", str(sr)]},
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "nopostoverwrites": False,
                "preferredcodec": "wav",
                "preferredquality": "0",
            },
            {
                "api": "https://sponsor.ajay.app",
                "categories": {
                    "interaction",
                    "music_offtopic",
                    "intro",
                    "outro",
                    "preview",
                    "selfpromo",
                    "sponsor",
                },
                "key": "SponsorBlock",
                "when": "after_filter",
            },
            {
                "force_keyframes": False,
                "key": "ModifyChapters",
                "remove_chapters_patterns": [],
                "remove_ranges": [],
                "remove_sponsor_segments": {
                    "interaction",
                    "music_offtopic",
                    "intro",
                    "outro",
                    "preview",
                    "selfpromo",
                    "sponsor",
                },
                "sponsorblock_chapter_title": "[SponsorBlock]: " "%(category_names)l",
            },
        ],
        "match_filter": save_metadata_fn,
        "cookiefile": cookie_file,
        "quiet": quiet,
        "retries": 25,
    }


def main(
    csv_path: Path,
    data_dir: Path,
    sample_rate: int = 16000,
    cookie_file: str = None,
    quiet=True,
) -> None:
    piano_id = csv_path.stem
    if (yt_id := get_yt_id(piano_id, data_dir)) == "":
        return
    output_file = data_dir / "audio" / f"{piano_id}.wav"
    if output_file.exists():
        print(f"{output_file} already downloaded")
        return
    meta_path = data_dir / "metadata" / f"{piano_id}.yaml"
    meta = OmegaConf.load(meta_path)
    meta.youtube = OmegaConf.create()
    meta.youtube.url = f"https://www.youtube.com/watch?v={yt_id}"
    save_metadata_fn = functools.partial(
        save_metadata,
        meta=meta,
        meta_path=meta_path,
    )
    yt_opts = get_opts(
        output_file,
        save_metadata_fn,
        sr=sample_rate,
        cookie_file=cookie_file,
        quiet=quiet,
    )

    try:
        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            ydl.download(meta.youtube.url)
            meta.youtube.duration = librosa.get_duration(path=str(output_file))
            OmegaConf.save(meta, meta_path)
    except yt_dlp.utils.DownloadError as e:
        print(e)
        print(f"{piano_id}: failed to download from {meta.youtube.url}")
        meta.pop("youtube")
        OmegaConf.save(meta, meta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    parser.add_argument(
        "--cookie_file",
        type=str,
        default=None,
        help="It is recommended to provide the cookie file for downloading videos with age restriction. More on https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=multiprocessing.cpu_count() // 2,
        help="Number of jobs in parallel",
    )
    parser.add_argument("--quiet", type=bool, default=True)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    (data_dir / "audio").mkdir(exist_ok=True)

    config = OmegaConf.load("config.yaml")
    sample_rate = config.dataset.sample_rate

    Parallel(n_jobs=args.n_jobs, backend="multiprocessing")(
        delayed(main)(
            csv_path,
            data_dir,
            sample_rate=sample_rate,
            cookie_file=args.cookie_file,
            quiet=args.quiet,
        )
        for csv_path in tqdm(list(data_dir.glob("youtube_csv/*.csv")))
    )
