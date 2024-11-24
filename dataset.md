# Build Dataset From Scratch

Install the Command-line tool of [dl-librescore](https://github.com/LibreScore/dl-librescore#command-line-tool).
Scrape MIDI files from Musescore.

```bash
python data/download_midi.py [data_dir]
python data/remove_invalid_midi.py [data_dir]
```

Search and download audio from YouTube.

```bash
python data/search_youtube.py [data_dir]
python data/download_youtube.py [data_dir]
```

Run this shell script or run the python scripts below manually.

```bash
chmod +x preprocess.sh
./preprocess.sh [data_dir]
```

Compute warp path and align MIDI with audio.
_Note: the output (synchronized) MIDI score will be transposed to match the key of the audio (if needed)._

```bash
python data/align_audio_midi.py [data_dir]
```

Convert midi file to numpy array.

```bash
python data/midi_to_numpy.py [data_dir]
```

Compute _WP-std_ and _beat flucuation_, then filter and split the dataset.

```bash
python data/compute_metrics.py [data_dir]
python data/generate_split.py [data_dir]
```
