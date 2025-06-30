# Data preprocessing pipeline

Compute warp path and align MIDI with audio.
_Note: the output (synchronized) MIDI score will be transposed to match the key of the audio (if needed)._

```bash
python data/align_audio_midi.py [data_dir]
```

Convert midi file to numpy array.

```bash
python data/midi_to_numpy.py [data_dir]
```

Compute _WP-std_ and _beat fluctuation_, then filter and split the dataset.

```bash
python data/compute_metrics.py [data_dir]
python data/generate_split.py [data_dir]
```
