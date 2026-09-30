"""Download daily adjusted closes from Yahoo Finance via yfinance.

Data is NOT committed: yfinance states that the Yahoo Finance API is intended
for personal use, so every user downloads locally. Fixed dates keep the window
reproducible; run_backtest.py records a SHA-256 of each file it reads.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yfinance as yf

DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]


def fetch(ticker: str, start: str, end: str, out_dir: Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if not df.empty:
            close = df["Close"].squeeze().rename("close")
            path = out_dir / f"{ticker.lower()}.csv"
            close.to_csv(path, index_label="date")
            print(f"{ticker}: {len(close)} rows, {close.index[0].date()} to {close.index[-1].date()}")
            return
        time.sleep(20 * attempt)
    raise SystemExit(f"No data returned for {ticker}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--start", default="2021-09-27")
    p.add_argument("--end", default="2026-09-26", help="exclusive")
    p.add_argument("--out-dir", default="data")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for t in a.tickers:
        fetch(t, a.start, a.end, out)


if __name__ == "__main__":
    main()
