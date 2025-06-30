import argparse
import copy
import io
import multiprocessing
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import librosa
import numpy as np
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from pretty_midi import PrettyMIDI
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import (
    compute_optimal_chroma_shift,
    make_path_strictly_monotonic,
    shift_chroma_vectors,
)
from synctoolbox.feature.chroma import (
    pitch_to_chroma,
    quantize_chroma,
    quantized_chroma_to_CENS,
)
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from tqdm import tqdm

warnings.filterwarnings(action="ignore")


def simple_adjust_times(
    midi_data: PrettyMIDI,
    original_times: np.ndarray,
    new_times: np.ndarray,
):
    """
    most of these codes are from original pretty_midi
    https://github.com/craffel/pretty-midi/blob/main/pretty_midi/pretty_midi.py
    """
    for instrument in midi_data.instruments:
        instrument.notes = [
            copy.deepcopy(note)
            for note in instrument.notes
            if note.start >= original_times[0] and note.end <= original_times[-1]
        ]
    # Get array of note-on locations and correct them
    note_ons = np.array(
        [
            note.start
            for instrument in midi_data.instruments
            for note in instrument.notes
        ]
    )
    adjusted_note_ons = np.interp(note_ons, original_times, new_times)
    # Same for note-offs
    note_offs = np.array(
        [note.end for instrument in midi_data.instruments for note in instrument.notes]
    )
    adjusted_note_offs = np.interp(note_offs, original_times, new_times)
    # Correct notes
    for n, note in enumerate(
        [note for instrument in midi_data.instruments for note in instrument.notes]
    ):
        note.start = (adjusted_note_ons[n] > 0) * adjusted_note_ons[n]
        note.end = (adjusted_note_offs[n] > 0) * adjusted_note_offs[n]
    # After performing alignment, some notes may have an end time which is
    # on or before the start time.  Remove these!
    midi_data.remove_invalid_notes()

    def adjust_events(event_getter):
        """This function calls event_getter with each instrument as the
        sole argument and adjusts the events which are returned."""
        # Sort the events by time
        for instrument in midi_data.instruments:
            event_getter(instrument).sort(key=lambda e: e.time)
        # Correct the events by interpolating
        event_times = np.array(
            [
                event.time
                for instrument in midi_data.instruments
                for event in event_getter(instrument)
            ]
        )
        adjusted_event_times = np.interp(event_times, original_times, new_times)
        for n, event in enumerate(
            [
                event
                for instrument in midi_data.instruments
                for event in event_getter(instrument)
            ]
        ):
            event.time = adjusted_event_times[n]
        for instrument in midi_data.instruments:
            # We want to keep only the final event which has time ==
            # new_times[0]
            valid_events = [
                event
                for event in event_getter(instrument)
                if event.time == new_times[0]
            ]
            if valid_events:
                valid_events = valid_events[-1:]
            # Otherwise only keep events within the new set of times
            valid_events.extend(
                event
                for event in event_getter(instrument)
                if event.time > new_times[0] and event.time < new_times[-1]
            )
            event_getter(instrument)[:] = valid_events

    # Correct pitch bends and control changes
    adjust_events(lambda i: i.pitch_bends)
    adjust_events(lambda i: i.control_changes)

    return midi_data


def get_features_from_audio(audio, tuning_offset, Fs, feature_rate, visualize=False):
    f_pitch = audio_to_pitch_features(
        f_audio=audio,
        Fs=Fs,
        tuning_offset=tuning_offset,
        feature_rate=feature_rate,
        verbose=visualize,
    )
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)

    f_pitch_onset = audio_to_pitch_onset_features(
        f_audio=audio,
        Fs=Fs,
        tuning_offset=tuning_offset,
        verbose=visualize,
    )
    f_DLNCO = pitch_onset_features_to_DLNCO(
        f_peaks=f_pitch_onset,
        feature_rate=feature_rate,
        feature_sequence_length=f_chroma_quantized.shape[1],
        visualize=visualize,
    )
    return f_chroma_quantized, f_DLNCO


def transpose_midi(midi_data: PrettyMIDI, shift: int) -> PrettyMIDI:
    for instrument in midi_data.instruments:
        instrument.notes = [copy.deepcopy(note) for note in instrument.notes]
        for note in instrument.notes:
            note.pitch += shift
    return midi_data


def compute_optimal_chroma_shift_wrapper(
    song_audio: np.ndarray, midi_audio: np.ndarray, sr: int, feature_rate: int
) -> int:
    """
    wrapper function for compute_optimal_chroma_shift

    if return != 0, the audio inputs are not in the same key
    """
    tuning_offset_song = librosa.estimate_tuning(y=song_audio, sr=sr) * 100
    tuning_offset_midi = librosa.estimate_tuning(y=midi_audio, sr=sr) * 100

    f_chroma_quantized_song, f_DLNCO_song = get_features_from_audio(
        song_audio,
        tuning_offset_song,
        Fs=sr,
        feature_rate=feature_rate,
        visualize=False,
    )
    f_chroma_quantized_midi, f_DLNCO_midi = get_features_from_audio(
        midi_audio,
        tuning_offset_midi,
        Fs=sr,
        feature_rate=feature_rate,
        visualize=False,
    )

    f_cens_1hz_song = quantized_chroma_to_CENS(
        f_chroma_quantized_song, 201, 50, feature_rate
    )[0]
    f_cens_1hz_midi = quantized_chroma_to_CENS(
        f_chroma_quantized_midi, 201, 50, feature_rate
    )[0]
    opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_song, f_cens_1hz_midi)
    return opt_chroma_shift


def get_warp_path(
    song_audio: np.ndarray,
    midi_audio: np.ndarray,
    sr: int,
    feature_rate: int = 50,
    strictly_monotonic: bool = True,
) -> tuple[np.ndarray, int]:
    """
    get warp path array with shape: [2, L].
    wp[0] = frames of song audio
    wp[1] = frames of midi

    code is adopted from https://github.com/meinardmueller/synctoolbox/blob/master/sync_audio_audio_full.ipynb
    """
    tuning_offset_song = librosa.estimate_tuning(y=song_audio, sr=sr) * 100
    tuning_offset_midi = librosa.estimate_tuning(y=midi_audio, sr=sr) * 100

    f_chroma_quantized_song, f_DLNCO_song = get_features_from_audio(
        song_audio,
        tuning_offset_song,
        Fs=sr,
        feature_rate=feature_rate,
        visualize=False,
    )
    f_chroma_quantized_midi, f_DLNCO_midi = get_features_from_audio(
        midi_audio,
        tuning_offset_midi,
        Fs=sr,
        feature_rate=feature_rate,
        visualize=False,
    )

    f_cens_1hz_song = quantized_chroma_to_CENS(
        f_chroma_quantized_song, 201, 50, feature_rate
    )[0]
    f_cens_1hz_midi = quantized_chroma_to_CENS(
        f_chroma_quantized_midi, 201, 50, feature_rate
    )[0]
    opt_chroma_shift = compute_optimal_chroma_shift(f_cens_1hz_song, f_cens_1hz_midi)

    f_chroma_quantized_midi = shift_chroma_vectors(
        f_chroma_quantized_midi, opt_chroma_shift
    )
    f_DLNCO_midi = shift_chroma_vectors(f_DLNCO_midi, opt_chroma_shift)

    wp = sync_via_mrmsdtw(
        f_chroma1=f_chroma_quantized_song,
        f_onset1=f_DLNCO_song,
        f_chroma2=f_chroma_quantized_midi,
        f_onset2=f_DLNCO_midi,
        input_feature_rate=feature_rate,
        step_weights=np.array([1.5, 1.5, 2.0]),
        threshold_rec=10**6,
        verbose=False,
    )

    if strictly_monotonic:
        wp = make_path_strictly_monotonic(wp)
    return wp / feature_rate, opt_chroma_shift


def main(
    meta_path: Path,
    data_dir: Path,
    sr: int,
    feature_rate: int,
):
    meta = OmegaConf.load(meta_path)
    piano_id = meta.piano.id
    song_path = data_dir / "audio" / f"{piano_id}.wav"
    midi_path = data_dir / "midi" / f"{piano_id}.mid"
    midi_transposed_path = data_dir / "midi_transposed" / f"{piano_id}.mid"
    midi_aligned_path = data_dir / "midi_aligned" / f"{piano_id}.mid"
    wp_path = data_dir / "warp_path" / f"{piano_id}.npy"
    beat_times_path = data_dir / "beat_times_aligned" / f"{piano_id}.npy"
    if wp_path.exists():
        print(f"{wp_path} already exists")
        return
    if not song_path.exists():
        print(f"{song_path} file not found")
        return

    song_audio, sr = librosa.load(str(song_path), sr=sr)
    song_audio = librosa.util.normalize(song_audio)
    midi_data = PrettyMIDI(str(midi_path))
    midi_audio = midi_data.synthesize(fs=sr)
    midi_audio = librosa.util.normalize(midi_audio)

    with redirect_stdout(io.StringIO()):
        opt_chroma_shift = compute_optimal_chroma_shift_wrapper(
            song_audio, midi_audio, sr, feature_rate
        )
        if opt_chroma_shift != 0:
            shift = (
                opt_chroma_shift
                if opt_chroma_shift <= abs(opt_chroma_shift - 12)
                else opt_chroma_shift - 12
            )
            midi_data = transpose_midi(midi_data, shift)
            midi_audio = midi_data.synthesize(fs=sr)
            midi_audio = librosa.util.normalize(midi_audio)

        wp, opt_chroma_shift = get_warp_path(
            song_audio,
            midi_audio,
            sr=sr,
            feature_rate=feature_rate,
        )

    midi_data.write(str(midi_transposed_path))
    beat_times = midi_data.get_beats()
    beat_times_aligned = np.interp(beat_times, wp[1], wp[0])
    midi_aligned = simple_adjust_times(midi_data, wp[1], wp[0])
    midi_aligned.write(str(midi_aligned_path))
    np.save(beat_times_path, beat_times_aligned)
    np.save(wp_path, wp)
    meta.piano.num_tracks = len(midi_data.instruments)
    meta.youtube.duration = librosa.get_duration(y=song_audio, sr=sr)
    meta.metrics = OmegaConf.create()
    meta.metrics.opt_chroma_shift = int(opt_chroma_shift)
    OmegaConf.save(meta, meta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    (data_dir / "midi_transposed").mkdir(exist_ok=True)
    (data_dir / "midi_aligned").mkdir(exist_ok=True)
    (data_dir / "warp_path").mkdir(exist_ok=True)
    (data_dir / "beat_times_aligned").mkdir(exist_ok=True)
    config = OmegaConf.load(args.config)
    feature_rate = config.dataset.dtw_feature_rate

    Parallel(n_jobs=multiprocessing.cpu_count() // 2, backend="multiprocessing")(
        delayed(main)(
            meta_path,
            data_dir,
            sr=22050,  # follow sync toolbox
            feature_rate=feature_rate,
        )
        for meta_path in tqdm(list(data_dir.glob("metadata/*.yaml")))
    )
