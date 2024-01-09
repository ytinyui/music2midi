# music2midi: Music to MIDI Piano Cover Generation

## Dependencies

- librosa==0.9.2
- pretty_midi==0.2.10
- pytorch==2.1.0
- pytorch-lightning==2.1.0
- transformers==4.34.0

## Conda environment

Install from environment.yaml

```bash
conda env create -f environment.yaml
conda activate music2midi
```

Install patched `note-seq` manually (fixed a bug)

```bash
git clone https://github.com/ytinyui/note-seq.git
cd note-seq
python setup.py install
```

## Dataset preparation

Install the Command-line tool of [dl-librescore](https://github.com/LibreScore/dl-librescore#command-line-tool)\
Download MIDI files from Musescore

```bash
python data/download_midi.py [data_dir]
python data/remove_invalid_midi.py [data_dir]
```

Search and download audio from YouTube

```bash
python data/search_youtube.py [data_dir]
python data/download_youtube.py [data_dir]
```

Run this shell script or run the python scripts below manually

```bash
chmod +x preprocess.sh
./preprocess.sh [data_dir]
```

Compute warp path (align MIDI with audio)
Note: the output (synchronized) MIDI score will be transposed to match the key of the audio (if needed)

```bash
python data/midi_song_align.py [data_dir]
```

Convert midi file to numpy array

```bash
python data/midi_to_numpy.py [data_dir]
python data/quantize_midi.py [data_dir]
```

Compute metrics, filter and split dataset

```bash
python -m data.compute_similarity [data_dir]
python data/compute_metrics.py [data_dir]
python data/generate_split.py [data_dir]
```

## Training

```bash
python train.py [data_dir]
```

## Evaluation

Evaluate the model based on melody F1 score

```bash
python evaluate.py [data_dir]
```
