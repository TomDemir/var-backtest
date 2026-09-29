"""Download SPY daily prices from Yahoo Finance via yfinance.

Data is NOT committed to the repository: Yahoo's data is for personal use
only (see yfinance's disclaimer), so each user downloads it locally.

Fixed dates make the window reproducible. Yahoo may still revise adjusted
prices after new dividends; the run script records a SHA-256 of the CSV so
any result can be traced to the exact file that produced it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yfinance as yf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2021-09-27")
    p.add_argument("--end", default="2026-09-26", help="exclusive")
    p.add_argument("--out", default="data/spy.csv")
    a = p.parse_args()

    df = yf.download(a.ticker, start=a.start, end=a.end, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"No data returned for {a.ticker}")
    close = df["Close"].squeeze().rename("close")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(a.out, index_label="date")
    print(f"{len(close)} rows, {close.index[0].date()} to {close.index[-1].date()} -> {a.out}")


if __name__ == "__main__":
    main()
