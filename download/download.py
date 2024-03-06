import multiprocessing
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from omegaconf import OmegaConf
import os
import argparse

parser = argparse.ArgumentParser(description="piano cover downloader")

parser.add_argument("dataset", type=str, default=None, help="provided csv")
parser.add_argument("output_dir", type=str, default=None, help="output dir")
parser.add_argument(
    "--num_audio",
    type=int,
    default=None,
    help="if specified, only {num_audio} pairs will be downloaded",
)
parser.add_argument("--parallel", default=multiprocessing.cpu_count(), type=int, help="number of parallel")
parser.add_argument("--dry_run", default=False, action="store_true", help="whether dry_run")


def youtube(url, 
            output_dir:Path, 
            output_file_template:str="%(id)s_-_%(uploader)s_-_%(title)s_-_%(duration)d_-_.%(ext)s",
            dry_run=False):
    output_template = output_dir / output_file_template
    return os.system(
        f"""yt-dlp -o "{str(output_template)}" \\
            --extract-audio \\
            --audio-quality 0 \\
            --audio-format wav \\
            --retries 25 \\
            {"--get-filename" if dry_run else ""}\\
            --prefer-ffmpeg \\
            --match-filter 'duration < 300 & duration > 150'\\
            --postprocessor-args "ffmpeg:-ac 1 -ar 16000" \\
            --quiet \\
            {url}"""
    )

def download_piano(piano_id:str, output_dir:Path, dry_run=False):
    if dry_run:
        youtube(url=f"https://www.youtube.com/watch?v={piano_id}",
        output_dir=output_dir,
        dry_run=dry_run)
        return

    if (output_dir / f"{piano_id}.wav").exists():
        print(piano_id, "file already exists (piano)")
        return
    
    youtube(url=f"https://www.youtube.com/watch?v={piano_id}",
        output_dir=output_dir,
        dry_run=dry_run)

    files = list(output_dir.glob(f"{piano_id}*_-_.wav"))
    if len(files) == 0:
        record_error(piano_id, "file not found (piano)")
        return
    if len(files) > 1:
        record_error(piano_id, "multiple files found (piano)")
        return

    file = files[0]
    ytid, uploader, title, duration, _ = file.stem.split("_-_")
    meta = OmegaConf.create()
    meta.piano = OmegaConf.create()
    meta.piano.ytid = ytid
    meta.piano.uploader = uploader
    meta.piano.title = title
    meta.piano.duration = int(duration)
    OmegaConf.save(meta, str(output_dir / f"{piano_id}.yaml"))
    file.rename(output_dir / f"{piano_id}.wav")


def download_song(piano_id:str, pop_id:str, output_dir:Path, dry_run=False):
    if dry_run:
        youtube(url=f"https://www.youtube.com/watch?v={pop_id}",
            output_dir=output_dir,
            dry_run=dry_run)
        return
    song_file = output_dir / piano_id / f"{pop_id}.wav"
    yaml_file = output_dir / f"{piano_id}.yaml"
    if (song_file).exists():
        print(pop_id, "file already exists (song)")
        return
    if (yaml_file).exists() == False:
        print(piano_id, "file not found (song)")
        return

    youtube(url=f"https://www.youtube.com/watch?v={pop_id}",
            output_dir=output_dir / piano_id,
            dry_run=dry_run)

    files = list((output_dir / piano_id).glob(f"{pop_id}*_-_.wav"))
    if len(files) == 0:
        record_error(pop_id, "file not found (song)")
        return
    if len(files) > 1:
        record_error(pop_id, "multiple files found (song)")
        return

    file = files[0]
    ytid, uploader, title, duration, _ = file.stem.split("_-_")
    meta = OmegaConf.load(str(yaml_file))
    meta.song = OmegaConf.create()
    meta.song.ytid = ytid
    meta.song.uploader = uploader
    meta.song.title = title
    meta.song.duration = int(duration)
    OmegaConf.save(meta, str(yaml_file))
    file.rename(song_file)

def record_error(id, msg):
    with open("error.csv", "a") as f:
        f.write(f"{id},{msg}\n")


if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df[: args.num_audio]

    print("Downloading piano audio:")
    Parallel(n_jobs=args.parallel)(
        delayed(download_piano)(piano_id, output_dir=Path(args.output_dir), dry_run=args.dry_run)
        for piano_id in tqdm(list(df["piano_ids"]))
        )
    print("Downloading song audio:")
    Parallel(n_jobs=args.parallel)(
        delayed(download_song)(piano_id, pop_id, output_dir=Path(args.output_dir), dry_run=args.dry_run)
        for piano_id, pop_id in tqdm(list(zip(df["piano_ids"], df["pop_ids"])))
    )
