import argparse
import json
import os
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
        total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_page_url(config: DictConfig, difficulty: str, genre: str, page_no: int):
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
    except (AttributeError, TypeError) as e:
        print(
            f"{e}. Metadata not found for {url}. You are probably blocked by Cloudflare."
        )

        raise
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
    max_retries: int = 5,
) -> int:
    """
    The program downloads the midi file to tmp_dir, then moves the file to the desired destination.

    Return:
    - 0 if the file is downloaded successfully
    - 1 if the file is not downloaded
    """
    retries = 0
    while True:
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
        )
        file = next(tmp_dir.glob("*.mid"), None)
        if file is not None:
            os.makedirs(output_path.parent, exist_ok=True)
            shutil.move(file, output_path)
            return 0
        else:
            retries += 1
            if retries > max_retries:
                with open("midi_not_downloaded.txt", "a") as f:
                    f.write(url + "\n")
                return 1


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
        "title": score_title,
        "duration": int(score_duration),
        "url": url,
        "view_count": content["config"]["statistic"]["scores_views"],
    }

    song_data = content["store"]["page"]["data"]["song"]
    song_title = song_data["name"]
    song_artist = song_data["artist"]["name"]
    song_info = {
        "title": song_title,
        "artist": song_artist,
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
    os.makedirs(midi_dir, exist_ok=True)
    os.makedirs(yaml_dir, exist_ok=True)

    metadata = get_metadata(url, session)
    score_meta = metadata["score"]
    if score_meta["duration"] < min_duration:
        return
    score_meta["difficulty"] = difficulty
    score_meta["genre"] = genre

    score_id = score_meta["id"]
    yaml_file = (yaml_dir / score_id).with_suffix(".yaml")
    if yaml_file.exists():
        return
    tmp_dir = tmp_dir / score_id
    os.makedirs(tmp_dir, exist_ok=True)
    output_code = download_midi(
        url,
        output_path=(midi_dir / score_id).with_suffix(".mid"),
        tmp_dir=tmp_dir,
    )
    os.rmdir(tmp_dir)
    if output_code == 0:
        OmegaConf.save(metadata, yaml_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str, default=None)
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="tmp",
        help="Temporary directory for downloading midi files",
    )
    args = parser.parse_args()

    config = OmegaConf.load("data/config.yaml")
    difficulty_genre_pairs = [
        (x, y) for x in config.code.difficulty.keys() for y in config.code.genre.keys()
    ]
    session = get_session()

    for difficulty, genre in (pbar := tqdm(difficulty_genre_pairs)):
        pbar.set_description(f"Downloading {genre} ({difficulty})")

        for page_no in tqdm(range(1, 101)):
            page_url = get_page_url(config, difficulty, genre, page_no)

            try:
                url_list = get_midi_url_list(page_url, session)
                if len(url_list) == 0:  # max page num exceeded
                    break
                Parallel(n_jobs=len(url_list), backend="threading")(
                    delayed(download_from_url)(
                        url,
                        session,
                        output_dir=Path(args.output_dir),
                        difficulty=difficulty,
                        genre=genre,
                        min_duration=config["min duration"],
                        tmp_dir=Path(args.tmp_dir),
                    )
                    for url in url_list
                )
            except Exception as e:
                print(e)
                print(
                    f"Failed to access {page_url}. This is probably a connection error. The program will stop."
                )
                exit(1)
