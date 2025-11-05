"""Scan Binance futures symbols for potential low-amplitude volume spikes."""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import aiohttp

from binance_utils import calculate_average_volume, is_green_volume_spike


BINANCE_FUTURES_API_URL = "https://fapi.binance.com"  # USDⓈ-M Futures endpoint.


@dataclass(slots=True)
class Detection:
    """Information about a detected volume spike."""

    symbol: str
    candle_open_time: dt.datetime
    detected_at: dt.datetime


async def fetch_trading_symbols(session: aiohttp.ClientSession) -> list[str]:
    """Return all USDⓈ-M perpetual symbols that are currently active on Binance."""

    async with session.get(f"{BINANCE_FUTURES_API_URL}/fapi/v1/exchangeInfo") as response:
        response.raise_for_status()
        payload = await response.json()

    symbols = [
        symbol_info["symbol"]
        for symbol_info in payload.get("symbols", [])
        if symbol_info.get("status") == "TRADING"
        and symbol_info.get("quoteAsset") == "USDT"
        and symbol_info.get("contractType") == "PERPETUAL"
    ]

    return sorted(symbols)


async def fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int,
) -> Sequence[Sequence[str]]:
    """Return the recent klines for a trading pair."""

    async with session.get(
        f"{BINANCE_FUTURES_API_URL}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    ) as response:
        response.raise_for_status()
        return await response.json()


async def analyse_symbol(
    session: aiohttp.ClientSession,
    symbol: str,
    *,
    interval: str,
    avg_window: int,
    volume_multiplier: float,
    max_body_ratio: float | None,
) -> Detection | None:
    """Analyse a trading pair and return a detection when it matches."""

    try:
        klines = await fetch_klines(session, symbol, interval, avg_window + 1)
    except aiohttp.ClientError as error:
        print(f"Failed to fetch klines for {symbol}: {error}", file=sys.stderr)
        return None

    if len(klines) < avg_window + 1:
        return None

    average_volume = calculate_average_volume(klines[:-1], avg_window)
    last_candle = klines[-1]

    if not is_green_volume_spike(
        last_candle,
        average_volume,
        volume_multiplier,
        max_body_ratio=max_body_ratio,
    ):
        return None

    open_time_ms = int(last_candle[0])
    open_time = dt.datetime.fromtimestamp(
        open_time_ms / 1000,
        tz=dt.timezone.utc,
    )

    return Detection(
        symbol=symbol,
        candle_open_time=open_time,
        detected_at=dt.datetime.now(dt.timezone.utc),
    )


async def detect_symbols(
    session: aiohttp.ClientSession,
    symbols: Iterable[str],
    *,
    interval: str,
    avg_window: int,
    volume_multiplier: float,
    max_body_ratio: float | None,
    max_concurrency: int,
) -> list[Detection]:
    """Return the list of symbols that satisfy the configured filters."""

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def _bounded(symbol: str) -> Detection | None:
        async with semaphore:
            return await analyse_symbol(
                session,
                symbol,
                interval=interval,
                avg_window=avg_window,
                volume_multiplier=volume_multiplier,
                max_body_ratio=max_body_ratio,
            )

    tasks = [asyncio.create_task(_bounded(symbol)) for symbol in symbols]
    results = await asyncio.gather(*tasks)

    return [result for result in results if result is not None]


def persist_detections(log_file: Path, detections: Sequence[Detection]) -> None:
    """Append detections to ``log_file`` in CSV format."""

    if not detections:
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not log_file.exists()

    with log_file.open("a", newline="") as file:
        writer = csv.writer(file)
        if is_new_file:
            writer.writerow(["symbol", "candle_open_time_utc", "detected_at_utc"])

        for detection in detections:
            writer.writerow(
                [
                    detection.symbol,
                    detection.candle_open_time.strftime("%Y-%m-%d %H:%M:%S"),
                    detection.detected_at.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Binance for green candles with abnormally high volume and a"
            " small body size."
        )
    )
    parser.add_argument(
        "--interval",
        default="1m",
        help="Kline interval to analyse (default: 1m)",
    )
    parser.add_argument(
        "--avg-window",
        type=int,
        default=60,
        help="Amount of candles used to compute the average volume (default: 60)",
    )
    parser.add_argument(
        "--volume-multiplier",
        type=float,
        default=3.0,
        help="Minimal multiplier between the latest volume and the average volume",
    )
    parser.add_argument(
        "--max-body-ratio",
        type=float,
        default=0.001,
        help=(
            "Maximum allowed candle body size relative to the open price. Use"
            " 0 or a negative value to disable this filter."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum amount of parallel requests to Binance (default: 10)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
        help=(
            "Seconds to wait between scans. Use 0 or a negative value to perform a"
            " single scan and exit (default: 60)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional path to append detections in CSV format",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv or sys.argv[1:])

    max_body_ratio = args.max_body_ratio if args.max_body_ratio > 0 else None
    poll_interval = args.poll_interval if args.poll_interval > 0 else 0.0

    async def _run() -> int:
        timeout = aiohttp.ClientTimeout(total=10)
        connector = aiohttp.TCPConnector(limit_per_host=args.max_concurrency)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            try:
                symbols = await fetch_trading_symbols(session)
            except aiohttp.ClientError as error:
                print(f"Unable to load trading symbols: {error}", file=sys.stderr)
                return 1

            last_reported: dict[str, dt.datetime] = {}

            while True:
                detections = await detect_symbols(
                    session,
                    symbols,
                    interval=args.interval,
                    avg_window=args.avg_window,
                    volume_multiplier=args.volume_multiplier,
                    max_body_ratio=max_body_ratio,
                    max_concurrency=args.max_concurrency,
                )

                new_detections: list[Detection] = []
                for detection in detections:
                    last_seen = last_reported.get(detection.symbol)
                    if last_seen is None or detection.candle_open_time > last_seen:
                        last_reported[detection.symbol] = detection.candle_open_time
                        new_detections.append(detection)

                print(
                    f"[{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC]"
                    f" Scanned {len(symbols)} symbols; found {len(new_detections)} new matches.",
                    flush=True,
                )

                for detection in new_detections:
                    print(
                        f"{detection.symbol} - {detection.candle_open_time:%Y-%m-%d %H:%M:%S} UTC",
                        flush=True,
                    )

                if args.log_file:
                    persist_detections(args.log_file, new_detections)

                if not new_detections:
                    print("No new symbols matched the configured filters.")

                if poll_interval == 0:
                    return 0

                try:
                    await asyncio.sleep(poll_interval)
                except asyncio.CancelledError:
                    raise

    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print("Stopping monitoring.")
        return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
