"""Volume analysis helpers."""

from __future__ import annotations

from typing import Iterable, Sequence


def extract_volumes(klines: Sequence[Sequence[str]]) -> list[float]:
    """Return the list of volumes from a kline payload.

    Parameters
    ----------
    klines:
        Sequence of kline payloads returned by the Binance REST API.  Each
        element is expected to be an iterable where the quote asset volume is
        stored at index 7 and provided as a string.
    """

    return [float(kline[7]) for kline in klines]


def calculate_average_volume(
    klines: Sequence[Sequence[str]] | Iterable[Sequence[str]],
    window: int,
) -> float:
    """Calculate the average traded volume for the provided klines.

    Parameters
    ----------
    klines:
        Sequence of kline payloads where element ``[7]`` contains the traded
        quote asset volume as a string.
    window:
        Amount of klines to use when computing the average.

    Returns
    -------
    float
        The arithmetic mean of the volumes in the last ``window`` klines.  If
        the provided iterable contains fewer than ``window`` elements the
        average of the entire iterable is returned.  When no klines are
        provided ``0.0`` is returned.
    """

    volumes = extract_volumes(list(klines)[-window:]) if window > 0 else []

    if not volumes:
        return 0.0

    return sum(volumes) / len(volumes)
