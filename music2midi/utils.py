import numpy as np
import pretty_midi


def numpy_to_midi(notes: np.ndarray) -> pretty_midi.PrettyMIDI:
    midi_data = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
    new_inst = pretty_midi.Instrument(program=0, name="Piano")

    new_inst.notes = [
        pretty_midi.Note(
            start=onset_time,
            end=offset_time,
            pitch=int(pitch),
            velocity=int(velocity),
        )
        for onset_time, offset_time, pitch, velocity in notes
    ]
    midi_data.instruments.append(new_inst)
    midi_data.remove_invalid_notes()
    return midi_data
