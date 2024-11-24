import collections

import bokeh
import bokeh.plotting
import note_seq
import numba
import numpy as np
import pandas as pd
import pretty_midi
from note_seq.protobuf import music_pb2

_CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class MIDIConversionError(Exception):
    pass


def piano_roll_to_instrument(piano_roll, fs=100, program=0, name=None):
    """Convert a Piano Roll array into a PrettyMidi instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    name: str
        The name of the instrument.

    Returns
    -------
    instrument : pretty_midi.Instrument
    """
    notes, frames = piano_roll.shape
    # pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, name=name)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], "constant")

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time,
            )
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0

    return instrument


@numba.njit()
def extract_melody_from_piano_roll(piano_roll: np.ndarray) -> np.ndarray:
    """
    return a piano roll of the melody that only has 12 pitch classes.

    input:
        piano_roll : (128, num_frames)

    return: (num_frames, )
    """
    num_frames = piano_roll.shape[1]
    ret = np.zeros((12, num_frames), dtype=np.int_)
    # for i in range(num_frames):
    #     if np.sum(piano_roll[:, i]) == 0:  # no onset at this frame
    #         continue
    #     (onset_pitches,) = np.nonzero(piano_roll[:, i])
    #     ret[i] = onset_pitches[-1] % 12

    # return ret
    num_frames = piano_roll.shape[1]
    for i in range(num_frames):
        if np.sum(piano_roll[:, i]) == 0:  # no onset at this frame
            continue
        (onsets,) = np.nonzero(piano_roll[:, i])
        ret[onsets[-1] % 12, i] = 1

    return ret


def evaluate_midi_result(
    target: pretty_midi.PrettyMIDI,
    predict: pretty_midi.PrettyMIDI,
    melody_only: bool = False,
):
    """
    target: pretty_midi.PrettyMIDI, the ground truth midi data
    predict: pretty_midi.PrettyMIDI, the predicted midi data

    return midi data with 3 instruments: TP, FN, FP
    """
    fs = 100
    target_end_time = target.get_end_time()
    predict_end_time = predict.get_end_time()
    end_time = max(predict_end_time, target_end_time)
    times = np.arange(0, end_time, 1 / fs)
    predict_piano_roll = predict.get_piano_roll(fs, times=times)
    target_piano_roll = target.get_piano_roll(fs, times=times)
    if melody_only:
        predict_piano_roll = extract_melody_from_piano_roll(predict_piano_roll)
        target_piano_roll = extract_melody_from_piano_roll(target_piano_roll)

    tp = np.logical_and(target_piano_roll, predict_piano_roll)
    fn = np.logical_and(target_piano_roll, np.logical_not(predict_piano_roll))
    fp = np.logical_and(np.logical_not(target_piano_roll), predict_piano_roll)
    tp_inst = piano_roll_to_instrument(tp, program=0, name="TP")
    fn_inst = piano_roll_to_instrument(fn, program=1, name="FN")
    fp_inst = piano_roll_to_instrument(fp, program=2, name="FP")

    pm = pretty_midi.PrettyMIDI()
    for inst in [tp_inst, fn_inst, fp_inst]:
        pm.instruments.append(inst)

    return pm


def plot_sequence(
    sequence,
    show_figure=True,
    width=1000,
    height=400,
    show_chords=False,
    melody_only=False,
):
    """Creates an interactive pianoroll for a NoteSequence.

    Example usage: plot a random melody.
      sequence = mm.Melody(np.random.randint(36, 72, 30)).to_sequence()
      plot_sequence(sequence)

    Args:
       sequence: A NoteSequence.
       show_figure: A boolean indicating whether or not to show the figure.
       width: An int indicating plot width in pixels. Default is 1000.
       height: An int indicating plot height in pixels. Default is 400.
       show_chords: If True, show chord changes on the x-axis.  Default is False.

    Returns:
       If show_figure is False, a Bokeh figure; otherwise None.
    """

    def _sequence_to_pandas_dataframe(sequence, note_width=0.4):
        """Generates a pandas dataframe from a sequence."""
        pd_dict = collections.defaultdict(list)
        for note in sequence.notes:
            pd_dict["start_time"].append(note.start_time)
            pd_dict["end_time"].append(note.end_time)
            pd_dict["duration"].append(note.end_time - note.start_time)
            pd_dict["pitch"].append(note.pitch)
            pd_dict["bottom"].append(note.pitch - note_width)
            pd_dict["top"].append(note.pitch + note_width)
            pd_dict["velocity"].append(note.velocity)
            pd_dict["fill_alpha"].append(note.velocity / 128.0)
            pd_dict["instrument"].append(note.instrument)
            pd_dict["program"].append(note.program)

        # If no velocity differences are found, set alpha to 1.0.
        if np.max(pd_dict["velocity"]) == np.min(pd_dict["velocity"]):
            pd_dict["fill_alpha"] = [1.0] * len(pd_dict["fill_alpha"])

        return pd.DataFrame(pd_dict)

    fig = bokeh.plotting.figure(tools="hover,box_zoom,reset,save")
    if width:
        fig.width = width
    if height:
        fig.height = height
    fig.xaxis.axis_label = "time (sec)"
    if melody_only:
        fig.yaxis.axis_label = "pitch class"
        # fig.yaxis.ticker =
        fig.yaxis.ticker = np.arange(0, 12)
        # fig.ygrid.ticker = bokeh.models.SingleIntervalTicker(interval=1)
    else:
        fig.yaxis.axis_label = "pitch"
        fig.yaxis.ticker = bokeh.models.SingleIntervalTicker(interval=12)
        fig.ygrid.ticker = bokeh.models.SingleIntervalTicker(interval=12)
    # Pick indexes that are maximally different in Spectral8 colormap.
    color_indexes = [0, 3, 5]

    if show_chords:
        chords = [
            (ta.time, str(ta.text))
            for ta in sequence.text_annotations
            if ta.annotation_type == _CHORD_SYMBOL
        ]
        fig.xaxis.ticker = bokeh.models.FixedTicker(ticks=[time for time, _ in chords])
        fig.xaxis.formatter = bokeh.models.CustomJSTickFormatter(
            code="""
        var chords = %s;
        return chords[tick];
    """
            % dict(chords)
        )

    # Create a Pandas dataframe and group it by instrument.
    note_width = 0.1 if melody_only else 0.4
    dataframe = _sequence_to_pandas_dataframe(sequence, note_width)
    instruments = sorted(set(dataframe["instrument"]))
    grouped_dataframe = dataframe.groupby("instrument")
    for counter, instrument in enumerate(instruments):
        instrument_df = grouped_dataframe.get_group(instrument)
        color_idx = color_indexes[counter % 3]
        color = bokeh.palettes.Blues[7][color_idx]  # 3 colors for TP, FN, FP
        source = bokeh.plotting.ColumnDataSource(instrument_df)
        fig.quad(
            top="top",
            bottom="bottom",
            left="start_time",
            right="end_time",
            line_color=color,
            fill_color=color,
            source=source,
            legend_label=sequence.instrument_infos[counter].name,
        )
    fig.xaxis.axis_label_text_font_size = "16pt"
    fig.yaxis.axis_label_text_font_size = "16pt"
    fig.legend.label_text_font_size = "14pt"
    fig.xaxis.major_label_text_font_size = "12pt"
    fig.yaxis.major_label_text_font_size = "12pt"
    fig.xaxis.axis_label_text_font_style = "normal"
    fig.yaxis.axis_label_text_font_style = "normal"
    fig.legend.location = "top_right"
    fig.select(dict(type=bokeh.models.HoverTool)).tooltips = (
        {  # pylint: disable=use-dict-literal
            "pitch": "@pitch",
            "program": "@program",
            "velo": "@velocity",
            "duration": "@duration",
            "start_time": "@start_time",
            "end_time": "@end_time",
        }
    )

    if show_figure:
        bokeh.plotting.output_notebook()
        bokeh.plotting.show(fig)

    return fig


def plot_midi_sequence(midi_data=None, midi_path=None):
    if midi_data is not None:
        if midi_path is not None:
            print("Both midi_data and midi_path are provided. Using midi_data.")
        return plot_sequence(note_seq.midi_to_note_sequence(midi_data))
    if midi_path is not None:
        return plot_sequence(note_seq.midi_file_to_note_sequence(midi_path))
    raise ValueError("Either midi_data or midi_path must be provided.")


def plot_midi_evaluation(
    target: pretty_midi.PrettyMIDI,
    predict: pretty_midi.PrettyMIDI,
    melody_only: bool = False,
) -> bokeh.plotting.figure:
    """
    Plot the evaluation result of midi data: TP, FN, FP.

    melody_only: bool, whether to only evaluate the melody part of the midi data
    """
    seq = note_seq.midi_to_note_sequence(
        evaluate_midi_result(target, predict, melody_only)
    )
    return plot_sequence(seq, melody_only=melody_only)
