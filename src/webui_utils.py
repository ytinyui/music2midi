import os
import shutil
import subprocess
from pathlib import Path

import yt_dlp


def post_process(video_path: Path, audio_path: Path):
    """
    merge piano audio to video to match duration then split it, otherwise the playback may not synchronize
    """
    output_dir = video_path.parent / "post-processed"
    output_dir.mkdir(exist_ok=True)
    merged_video_path = output_dir / "merged.mp4"
    output_video_path = output_dir / video_path.name
    output_audio_path = output_dir / audio_path.name

    merge_process = subprocess.call(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0",
            "-map",
            "1",
            "-c",
            "copy",
            str(merged_video_path),
            "-y",
        ]
    )
    assert merge_process == 0

    video_stream_args = (
        ["-map", "0:v"] if video_stream_present(merged_video_path) else []
    )
    split_process = subprocess.call(
        ["ffmpeg", "-loglevel", "error", "-i", str(merged_video_path)]
        + video_stream_args
        + [
            "-map",
            "0:a:0",
            "-c",
            "copy",
            str(output_video_path),
            "-y",
            "-map",
            "0:a:1",
            str(output_audio_path),
            "-y",
        ],
    )
    assert split_process == 0

    os.rename(output_video_path, video_path)
    os.rename(output_audio_path, audio_path)
    shutil.rmtree(output_dir)


def download_video(url: str, audio_path: Path):
    ydl_opts = {
        "format_sort": ["res:720"],
        "merge_output_format": "mp4",
        "noprogress": True,
        "outtmpl": {"default": str(audio_path)},
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)


def video_stream_present(file_path: Path):
    """
    return: True if video stream is found in the input file, False otherwise
    """
    return not subprocess.call(
        [
            "ffmpeg",
            "-loglevel",
            "panic",
            "-i",
            str(file_path),
            "-map",
            "v",
            "-vframes",
            "1",
            "-c",
            "copy",
            "-f",
            "null",
            "-",
        ]
    )
