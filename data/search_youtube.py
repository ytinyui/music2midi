import argparse
import re
from pathlib import Path
from typing import Union

import yt_dlp
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def valid_title(meta: DictConfig) -> bool:
    for title in [meta.score.title, meta.song.title]:
        for word in ["misc", "medley", "mashup", "mashups", "guess"]:
            if re.search(rf"\b{word}\b", title, re.IGNORECASE):
                return False
    return True


def valid_artist(meta: DictConfig) -> bool:
    artist = meta.song.artist.lower()
    if artist == "misc" or "medley" in artist:
        return False
    return True


def get_search_key(meta: DictConfig, filter_words: list[str] = []) -> str:
    if "misc" in meta.song.artist.lower() or "OST" in meta.song.title:
        search_key = meta.score.title
    else:
        search_key = f"{meta.song.artist} - {meta.song.title}"

    for x in filter_words:
        search_key = search_key.casefold().replace(x.casefold(), " ")

    return search_key.strip()


def search_youtube(
    meta: DictConfig,
    output_dir: Path,
    cookie_file: str = None,
    filter_words: list[str] = [],
    search_count: int = 3,
    quiet: bool = True,
) -> None:
    search_key = get_search_key(meta, filter_words)
    score_id = meta.score.id
    score_duration = meta.score.duration

    output_csv = output_dir / (score_id + ".csv")
    with output_csv.open("w") as f:
        f.write(
            "score_id,yt_id,search_key,yt_title,score_duration,yt_duration,yt_view_count\n"
        )

    def download_filter(
        info: dict,
        score_duration: int = score_duration,
    ) -> Union[str, None]:
        """
        return str: filtered
        return None: accepted
        """
        if (duration := info.get("duration")) is None:
            return f"{score_id} | no duration"
        if abs(duration - score_duration) > 60:
            return f"{score_id} | duration filtered: {duration}"

        for x in filter_words:
            if x.casefold() in (title := info.get("title")).casefold():
                return f"{score_id} | title filtered: {title}"

        with output_csv.open("a") as f:
            f.write(
                "{},{},{},{},{},{},{}\n".format(
                    score_id,
                    info.get("id"),
                    search_key.replace(",", " "),
                    info.get("title").replace(",", " ").replace("\n", "").strip(),
                    score_duration,
                    info.get("duration"),
                    info.get("view_count"),
                )
            )
        return None

    opts = {
        "match_filter": download_filter,
        "cookiefile": cookie_file,
        "ignore_no_formats_error": True,
        "noprogress": True,
        "quiet": quiet,
        "simulate": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(f"ytsearch{search_count}:{search_key}")


def main(
    meta_path: Path,
    output_dir: Path,
    cookie_file: str,
    filter_words: list[str] = [],
    quiet: bool = True,
):
    meta = OmegaConf.load(meta_path)
    if not all(
        [
            valid_title(meta),
            valid_artist(meta),
            meta.score.duration <= 600,
        ]
    ):
        return
    search_youtube(
        meta,
        output_dir,
        cookie_file=cookie_file,
        filter_words=filter_words,
        quiet=quiet,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    parser.add_argument(
        "--cookie_file",
        type=str,
        default=None,
        help="Optional. Metadata of videos with age restriction cam be fetched without logging in. More on https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=30,
        help="Number of jobs in parallel",
    )
    parser.add_argument("--quiet", type=bool, default=True)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "youtube_csv"
    output_dir.mkdir(exist_ok=True)

    filter_words = [
        "piano",
        "transcription",
        "covered",
        "solo",
        "medley",
        "mashup",
        "guess",
        "synthesia",
        "tutorial",
        "鋼琴",
        "弾いてみた",
        "ピアノ",
    ]

    Parallel(n_jobs=args.n_jobs)(
        delayed(main)(
            meta_path,
            output_dir,
            cookie_file=args.cookie_file,
            filter_words=filter_words,
            quiet=args.quiet,
        )
        for meta_path in tqdm(list(data_dir.glob("metadata/*.yaml")))
    )
