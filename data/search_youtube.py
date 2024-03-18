import argparse
import re
from pathlib import Path
from typing import Optional, Union

import yt_dlp
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def valid_title(meta: DictConfig) -> bool:
    for title in [meta.piano.title, meta.song.title]:
        for word in ["misc", "medley", "mashup", "mash-up", "mashups", "guess"]:
            if re.search(rf"\b{word}\b", title, re.IGNORECASE):
                return False
    return True


def valid_artist(meta: DictConfig) -> bool:
    artist = meta.song.artist.lower()
    for word in ["medley", "mashup", "mash-up", "mashups", "guess"]:
        if re.search(rf"\b{word}\b", artist, re.IGNORECASE):
            return False
    return True


def get_search_key(meta: DictConfig, word_blacklist: list[str] = []) -> str:
    if (
        "misc" in meta.song.artist.lower()
        or "anonymous" in meta.song.artist.lower()
        or "OST" in meta.song.title
    ):
        search_key = meta.piano.title
    else:
        search_key = f"{meta.song.artist} - {meta.song.title}"

    for x in word_blacklist:
        search_key = search_key.casefold().replace(x.casefold(), " ")

    return search_key.strip()


def search_youtube(
    meta: DictConfig,
    output_dir: Path,
    word_blacklist: list[str],
    cookie_file: str = None,
    search_count: int = 3,
    quiet: bool = True,
) -> None:
    search_key = get_search_key(meta, word_blacklist)
    piano_id = meta.piano.id
    score_duration = meta.piano.duration

    output_csv = output_dir / f"{piano_id}.csv"
    if output_csv.exists():
        print(f"{output_csv} already exists")
        return
    with output_csv.open("w") as f:
        f.write(
            "piano_id,yt_id,search_key,yt_title,score_duration,yt_duration,yt_view_count\n"
        )

    def download_filter(
        info: dict,
        score_duration: int = score_duration,
    ) -> Union[str, None]:
        """
        return str: discarded
        return None: accepted
        """
        if (duration := info.get("duration")) is None:
            return f"{piano_id} | no duration"
        if abs(duration - score_duration) > 60:
            return f"{piano_id} | duration filtered: {duration}"

        for x in word_blacklist:
            if x.casefold() in (title := info.get("title")).casefold():
                return f"{piano_id} | title filtered: {title}"

        with output_csv.open("a") as f:
            f.write(
                "{},{},{},{},{},{},{}\n".format(
                    piano_id,
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
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(f"ytsearch{search_count}:{search_key}")
    except Exception as e:
        print(f"{e} ({piano_id})")
        output_csv.unlink()
        raise


def main(
    meta_path: Path,
    output_dir: Path,
    cookie_file: str,
    word_blacklist: Optional[list[str]] = None,
    quiet: bool = True,
):
    if word_blacklist is None:
        word_blacklist = []
    meta = OmegaConf.load(meta_path)
    if not all(
        [
            valid_title(meta),
            valid_artist(meta),
            meta.piano.duration <= 600,
        ]
    ):
        return
    if meta.piano.genre != "classical":
        word_blacklist.append("piano")
    search_youtube(
        meta,
        output_dir,
        word_blacklist,
        cookie_file=cookie_file,
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

    word_blacklist = [
        "transcription",
        "cover",
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
            word_blacklist=word_blacklist,
            quiet=args.quiet,
        )
        for meta_path in tqdm(list(data_dir.glob("metadata/*.yaml")))
    )
