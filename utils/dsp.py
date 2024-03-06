import numpy as np


def pad_audio(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the shorter audio to the longer one.
    Returns a tuple of the two arrays.
    """
    return (
        np.pad(x, (0, max(len(y) - len(x), 0))),
        np.pad(y, (0, max(len(x) - len(y), 0))),
    )


def to_stereo(x, y):
    x, y = pad_audio(x, y)
    return np.stack((x, y))
