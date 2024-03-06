import argparse
import copy
import html
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Cache-Control": "max-age=0",
            "Upgrade-Insecure-Requests": "1",
        }
    )
    retry_strategy = Retry(
        total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_page_url(config: DictConfig, difficulty: str, genre: str, page_no: int) -> str:
    assert type(page_no) is int and page_no > 0
    return "https://musescore.com/sheetmusic?type=non-official&sort=viewcount&instrument=2&instrumentation=114&complexity={}&genres={}&page={}".format(
        config.code.difficulty[difficulty], config.code.genre[genre], page_no
    )


def get_content(url: str, session: requests.Session) -> dict:
    time.sleep(1)
    response = session.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    try:
        div = soup.find("div", {"class": re.compile("js-(?!page)")})
        key = next((k for k, v in div.attrs.items() if "data-" in k), None)
        content = json.loads(div.get(key))
    except (AttributeError, TypeError):
        raise ConnectionRefusedError("Blocked by Cloudflare")
    return content


def get_midi_url_list(page_url: str, session: requests.Session) -> list[str]:
    """
    Return: list of urls on the webpage.
    """
    content = get_content(page_url, session)
    scores = content["store"]["page"]["data"]["scores"]
    return [x["href"] for x in scores]


def download_midi(
    url: str,
    output_path: Path,
    tmp_dir: Path,
    max_retries: int = 3,
) -> None:
    """
    Download the midi file to tmp_dir, then move the file to the desired destination and remove tmp_dir
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    retries = 0
    while retries < max_retries:
        subprocess.run(
            [
                "python",
                "data/librescore.py",
                url,
                "--output_dir",
                str(tmp_dir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        file = next(tmp_dir.glob("*.mid"), None)
        if file is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(file, output_path)
            tmp_dir.rmdir()
            return
        retries += 1
    raise FileNotFoundError(f"MIDI File {output_path} not downloaded.")


def get_metadata(url: str, session: requests.Session) -> dict:
    content = get_content(url, session)
    score_data = content["store"]["score"]
    score_id = score_data["id"]
    score_title = score_data["title"]
    score_duration = content["store"]["jmuse_settings"]["score_player"]["json"][
        "metadata"
    ]["duration"]
    score_info = {
        "id": str(score_id),
        "title": html.unescape(score_title),
        "duration": int(score_duration),
        "url": url,
        "view_count": content["config"]["statistic"]["scores_views"],
    }

    song_data = content["store"]["page"]["data"]["song"]
    song_title = song_data["name"]
    song_artist = song_data["artist"]["name"]
    song_info = {
        "title": html.unescape(song_title),
        "artist": html.unescape(song_artist),
    }

    return {"score": score_info, "song": song_info}


def download_from_url(
    url: str,
    session: requests.Session,
    output_dir: Path,
    difficulty: Path,
    genre: str,
    min_duration: int,
    tmp_dir: Path,
) -> None:
    midi_dir = output_dir / "midi"
    yaml_dir = output_dir / "metadata"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    score_id = url.split("/")[-1]
    yaml_file = (yaml_dir / score_id).with_suffix(".yaml")
    if yaml_file.exists():
        return
    try:
        metadata = get_metadata(url, session)
    except KeyError:  # webpage not available
        return
    score_meta = metadata["score"]
    assert score_id == score_meta["id"]
    if score_meta["duration"] < min_duration:
        return
    score_meta["difficulty"] = difficulty
    score_meta["genre"] = genre
    tmp_dir = tmp_dir / score_id
    download_midi(
        url,
        output_path=(midi_dir / score_id).with_suffix(".mid"),
        tmp_dir=tmp_dir,
    )
    OmegaConf.save(metadata, yaml_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str, default=None)
    parser.add_argument("--log_file", type=str, default="progress.json")
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="tmp",
        help="Temporary directory for downloading midi files",
    )
    args = parser.parse_args()
    config = OmegaConf.load("data/config.yaml")
    # read log file, create one if does not exist
    log_file = Path(args.log_file)
    if log_file.exists():
        with log_file.open("r") as f:
            progress = json.load(f)
    else:
        tmp_dict = {}
        for difficulty in config.code.difficulty.keys():
            tmp_dict[difficulty] = 1
        progress = {}
        for genre in config.code.genre.keys():
            progress[genre] = copy.copy(tmp_dict)
        with log_file.open("w") as f:
            json.dump(progress, f, indent=2)

    difficulty_genre_pairs = [
        (y, x) for x in config.code.difficulty.keys() for y in config.code.genre.keys()
    ]
    session = get_session()

    for genre, difficulty in tqdm(difficulty_genre_pairs):
        pbar = tqdm(range(progress[genre][difficulty], 101), leave=False)
        pbar.set_description(f"Downloading {genre} ({difficulty})")
        for page_no in pbar:
            page_url = get_page_url(config, difficulty, genre, page_no)
            try:
                url_list = get_midi_url_list(page_url, session)
                if len(url_list) == 0:  # max page num exceeded
                    break
                Parallel(n_jobs=len(url_list) // 2)(
                    delayed(download_from_url)(
                        url,
                        session,
                        output_dir=Path(args.output_dir),
                        difficulty=difficulty,
                        genre=genre,
                        min_duration=config["min_duration"],
                        tmp_dir=Path(args.tmp_dir),
                    )
                    for url in url_list
                )
            except Exception as e:
                print(f"{e}. Error occurred while accessing {page_url}.")
                raise

            progress[genre][difficulty] = page_no + 1
            with log_file.open("w") as f:
                json.dump(progress, f, indent=2)
