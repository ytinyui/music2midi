# Music2MIDI: Pop Music to MIDI Piano Cover Generation

Music2MIDI: a deep learning model that generates MIDI piano cover songs from the original song audio.
Official repository for [Music2MIDI: Pop Music to MIDI Piano Cover Generation](https://dl.acm.org/doi/10.1007/978-981-96-2064-7_8).

## Setup

### FFmpeg

Install [FFmpeg](https://www.ffmpeg.org/download.html).

### FluidSynth

Install [FluidSynth](https://github.com/FluidSynth/fluidsynth/wiki/Download).

### Conda Environment

```bash
conda env create -f environment.yaml
conda activate music2midi
```

### Model Checkpoint

Download trained model checkpoint in Releases.

## Usage

### Web UI

```python
python webui.py --ckpt [ckpt_path]
```

### Python API

See demo.ipynb.

## Dataset preparation

Download the dataset in Releases then unzip it.

Download the audio files from YouTube.

```python
python data/download_youtube.py [data_dir]
```

### Data preprocessing

The released dataset is already preprocessed.
See dataset.md if you want to prepare your own dataset.

## Training

```bash
python train.py [data_dir]
```

## Evaluation

Evaluate the model based on melody chroma accuracy.

```bash
python evaluate.py [data_dir] --ckpt [ckpt_path]
```

## Troubleshooting

- If you encounter the error `module 'soundfile' has no attribute 'SoundFileRuntimeError'`, uninstall `soundfile` then install it again:

  ```bash
  pip uninstall pysoundfile
  pip uninstall soundfile
  pip install soundfile
  ```

- If you encounter problems with FluidSynth like `/usr/lib/libinstpatch-1.0.so.2: undefined symbol: g_once_init_leave_pointer`, consider downgrading `libinstpatch` in your system.
