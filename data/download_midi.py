import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
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
    return "https://musescore.com/sheetmusic?sort=rating&instrument=2&instrumentation=114&complexity={}&genres={}&page={}".format(
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
    except TypeError as e:
        raise e(
            f"Metadata not found for {url}. You are probably blocked by Cloudflare."
        )
    return content


def get_midi_url_list(page_url: str, session: requests.Session) -> list[str]:
    content = get_content(page_url, session)
    scores = content["store"]["page"]["data"]["scores"]
    return [x["href"] for x in scores]


def download_midi(
    url: str, filename: str, output_dir: str, tmp_dir="tmp", max_retries: int = 5
) -> int:
    retries = 0
    while True:
        subprocess.run(
            ["python", "data/librescore.py", url, "--output_dir", tmp_dir],
            capture_output=True,
            text=True,
        )
        file = next(Path(tmp_dir).glob("*.mid"), None)
        if file is not None:
            os.makedirs(output_dir, exist_ok=True)
            file.rename(Path(output_dir) / filename)
            return 0
        else:
            retries += 1
            if retries > max_retries:
                with open("midi_not_downloaded.txt", "a") as f:
                    f.write(url + "\n")
                return 1
            print(
                "Midi file {} from {} not downloaded. Retrying... (attempt {}/{})".format(
                    filename, url, retries, max_retries
                )
            )


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
    }

    song_data = content["store"]["page"]["data"]["song"]
    song_title = song_data["name"]
    song_artist = song_data["artist"]["name"]
    song_info = {
        "title": song_title,
        "artist": song_artist,
    }

    return {"score": score_info, "song": song_info}


def download_from_url_list(
    url_list: list,
    session: requests.Session,
    output_dir: str,
    difficulty: str,
    genre: str,
    min_duration: int,
    tmp_dir: str,
) -> int:
    """
    return: number of files successfully downloaded
    """
    midi_dir = Path(output_dir) / "midi"
    yaml_dir = Path(output_dir) / "metadata"
    midi_dir.mkdir(exist_ok=True)
    yaml_dir.mkdir(exist_ok=True)

    result = []
    for url in url_list:
        metadata = get_metadata(url, session)
        score_meta = metadata["score"]
        if score_meta["duration"] < min_duration:
            continue
        score_meta["difficulty"] = difficulty
        score_meta["genre"] = genre

        output_no = download_midi(
            url,
            filename=score_meta["id"] + ".mid",
            output_dir=midi_dir,
            tmp_dir=tmp_dir,
        )
        result.append(output_no)
        if output_no == 0:
            yaml_file = (yaml_dir / score_meta["id"]).with_suffix(".yaml")
            OmegaConf.save(metadata, yaml_file)

    return result.count(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str, default=None)
    parser.add_argument("--tmp_dir", type=str, default="tmp")
    parser.add_argument("--resume_url", type=str, default=None)
    args = parser.parse_args()

    config = OmegaConf.load("data/config.yaml")
    resumed = 0 if args.resume_url is not None else 1
    difficulty_genre_pairs = [
        (x, y) for x in config.code.difficulty.keys() for y in config.code.genre.keys()
    ]
    session = get_session()

    for difficulty, genre in tqdm(difficulty_genre_pairs):
        num_files_required = config["num scores"][difficulty][genre]
        progress_bar = tqdm(total=num_files_required)
        progress_bar.set_description(f"Downloading genre: {genre} ({difficulty})")

        num_files = 0
        page_no = 1
        while num_files < num_files_required and page_no <= 100:
            page_url = get_page_url(config, difficulty, genre, page_no)
            # handle resuming if resume_url is provided
            if resumed == 0 and page_url != args.resume_url:
                page_no += 1
                if page_no > 100:
                    break
                continue
            if resumed == 0 and page_url == args.resume_url:
                resumed = 1
                print(f"\nProgress resumed at {page_url}")

            try:
                url_list = get_midi_url_list(page_url, session)
                download_from_url_list(
                    url_list,
                    session,
                    output_dir=args.output_dir,
                    difficulty=difficulty,
                    genre=genre,
                    min_duration=config["min duration"],
                    tmp_dir=args.tmp_dir,
                )
            except TimeoutError as e:
                print(f"{e}, skipping {page_url}.")
            except Exception as e:
                print(
                    f"{e}, Failed to download from {page_url}. You may resume by providing the url for --resume_url."
                )
                exit(1)

            yaml_files = [
                OmegaConf.load(file)
                for file in Path(args.output_dir).glob("metadata/*.yaml")
            ]
            num_files = sum(
                1
                for config in yaml_files
                if config.score["difficulty"] == difficulty
                and config.score["genre"] == genre
            )
            progress_bar.update(num_files - progress_bar.n)
            page_no += 1
