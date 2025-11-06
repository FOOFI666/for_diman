"""Volume analysis helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum


class InsufficientKlineDataError(ValueError):
    """Raised when there is not enough historical data to compute a metric."""


class VolumeField(Enum):
    """Enum describing where a particular volume metric lives in a kline."""

    BASE = ("base", 5)
    QUOTE = ("quote", 7)

    @property
    def label(self) -> str:
        """Human readable label for the field."""

        return self.value[0]

    @property
    def index(self) -> int:
        """Return the position of the volume metric inside a kline payload."""

        return self.value[1]

    @classmethod
    def from_label(cls, label: str) -> "VolumeField":
        """Return the enum member matching ``label`` (case-insensitive)."""

        normalized = label.strip().lower()
        for field in cls:
            if field.label == normalized:
                return field
        msg = f"Unknown volume field: {label!r}. Expected one of: base, quote."
        raise ValueError(msg)


def extract_volumes(
    klines: Sequence[Sequence[str]], *, volume_field: VolumeField = VolumeField.QUOTE
) -> list[float]:
    """Return the list of volumes from a kline payload.

    Parameters
    ----------
    klines:
        Sequence of kline payloads returned by the Binance REST API.
    volume_field:
        Field describing whether to extract the base asset volume (index 5) or
        the quote asset volume (index 7).
    """

    volume_index = volume_field.index
    return [float(kline[volume_index]) for kline in klines]


def calculate_average_volume(
    klines: Sequence[Sequence[str]] | Iterable[Sequence[str]],
    window: int,
    *,
    volume_field: VolumeField = VolumeField.QUOTE,
) -> float:
    """Calculate the average traded volume for the provided klines.

    Parameters
    ----------
    klines:
        Sequence of kline payloads containing Binance volume information.
    window:
        Amount of klines to use when computing the average. Must be a positive
        integer.
    volume_field:
        Field describing which Binance kline volume metric should be used for
        the calculation.

    Returns
    -------
    float
        The arithmetic mean of the volumes in the last ``window`` klines.

    Raises
    ------
    ValueError
        If ``window`` is not a positive integer.
    InsufficientKlineDataError
        If fewer than ``window`` klines are provided.
    """

    if window <= 0:
        raise ValueError("Window size must be a positive integer")

    kline_list = list(klines)
    if len(kline_list) < window:
        raise InsufficientKlineDataError(
            f"Expected at least {window} klines but received {len(kline_list)}"
        )

    volumes = extract_volumes(kline_list[-window:], volume_field=volume_field)
    return sum(volumes) / window
