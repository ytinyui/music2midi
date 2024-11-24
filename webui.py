import argparse
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import soundfile as sf
from flask import Flask, render_template, request

import music2midi.webui_utils as utils
from music2midi.model import Music2MIDI

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = Path("static/uploads")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    url = request.form.get("url")
    upload_file = request.files.get("file")
    if url == upload_file.filename == "":
        return render_template(
            "index.html", error="Please provide a URL or upload a file."
        )
    if upload_file.filename != "":
        result_dir = app.config["UPLOAD_FOLDER"] / "local" / upload_file.filename
    else:
        parsed_url = urlparse(url)
        try:
            song_id = parse_qs(parsed_url.query)["v"][0]
        except KeyError:
            song_id = url
        result_dir = app.config["UPLOAD_FOLDER"] / "youtube" / song_id

    result_dir.mkdir(parents=True, exist_ok=True)
    audio_path = result_dir / "output.mp3"
    video_path = result_dir / "input.mp4"
    midi_path = result_dir / "output.mid"
    # return results directly if the files already exist
    if audio_path.exists() and video_path.exists():
        print("Using existing result at", str(result_dir))
        return render_template(
            "result.html",
            video_path=str(video_path),
            audio_path=str(audio_path),
            display_video=utils.video_stream_present(video_path),
        )
    try:
        if upload_file.filename != "":
            print("Using uploaded file", upload_file.filename)
            upload_file.save(video_path)
        else:
            print("Downloading video from", url)
            utils.download_video(url, video_path)

        print("Generating result")
        midi_data = model.generate(video_path)
        midi_data.write(str(midi_path))
        print("MIDI file is written to", str(midi_path))
        # post-process
        sr = 48000
        audio_y = midi_data.fluidsynth(fs=sr)
        sf.write(str(audio_path), audio_y, sr)
        print("Post-processing")
        utils.post_process(video_path, audio_path)
    except:
        shutil.rmtree(result_dir)
        raise

    return render_template(
        "result.html",
        video_path=str(video_path),
        audio_path=str(audio_path),
        display_video=utils.video_stream_present(video_path),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="model checkpoint path")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="model config path"
    )
    args = parser.parse_args()

    model = Music2MIDI.load_from_checkpoint(args.ckpt, config_path=args.config)
    model = model.cuda().eval()
    print("Model loaded successfully")
    app.run(port=5736)
