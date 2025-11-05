"""Detection helpers for finding promising Binance candles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CandleCharacteristics:
    """Normalized metrics that describe a single kline/candle."""

    is_green: bool
    volume: float
    body_ratio: float


def _parse_candle(kline: list[str] | tuple[str, ...]) -> CandleCharacteristics:
    open_price = float(kline[1])
    high_price = float(kline[2])
    low_price = float(kline[3])
    close_price = float(kline[4])
    volume = float(kline[5])

    price_reference = open_price if open_price != 0 else 1.0
    body_ratio = abs(close_price - open_price) / price_reference

    return CandleCharacteristics(
        is_green=close_price > open_price,
        volume=volume,
        body_ratio=body_ratio,
    )


def is_green_volume_spike(
    kline: list[str] | tuple[str, ...],
    average_volume: float,
    volume_multiplier: float,
    *,
    max_body_ratio: float | None = None,
) -> bool:
    """Return ``True`` when the candle satisfies the configured filters.

    Parameters
    ----------
    kline:
        The candle to analyse in Binance kline format.
    average_volume:
        Average volume computed over the desired look-back window.
    volume_multiplier:
        Multiplier that the candle volume must exceed.
    max_body_ratio:
        Optional limit for the candle body expressed as a ratio relative to the
        opening price.  When ``None`` the body size is not taken into account.
    """

    candle = _parse_candle(kline)

    if not candle.is_green:
        return False

    if volume_multiplier <= 0:
        return False

    if candle.volume < average_volume * volume_multiplier:
        return False

    if max_body_ratio is not None and candle.body_ratio > max_body_ratio:
        return False

    return True
