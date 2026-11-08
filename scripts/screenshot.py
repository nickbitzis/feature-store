"""Capture README screenshots with headless Chrome. Start the server first: make run.

Usage: uv run python scripts/screenshot.py --port 8858
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
]
DOCS = Path(__file__).resolve().parents[1] / "docs"


def chrome() -> str:
    for c in CHROME_PATHS:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("Chrome not found")


def shoot(url: str, out: Path, width: int, height: int, dark: bool = False) -> None:
    args = [
        chrome(),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        "--virtual-time-budget=8000",
        f"--screenshot={out}",
        "--force-dark-mode" if dark else "--disable-features=WebContentsForceDark",
        f"--blink-settings=preferredColorScheme={0 if dark else 1}",
    ]
    subprocess.run([*args, url], check=True, capture_output=True, timeout=90)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8858)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    with urllib.request.urlopen(f"{base}/healthz") as r:
        json.load(r)
    with urllib.request.urlopen(f"{base}/cards/examples") as r:
        card = json.load(r)["card_testing"][0]
    DOCS.mkdir(exist_ok=True)
    shoot(f"{base}/#/model", DOCS / "screenshot.png", 1400, 1250)
    shoot(f"{base}/#/timeline/{card}", DOCS / "timeline.png", 1400, 1500)
    shoot(f"{base}/#/skew/on", DOCS / "skew.png", 1400, 1000)
    shoot(f"{base}/#/registry", DOCS / "registry.png", 1400, 1300)
    shoot(f"{base}/#/latency", DOCS / "latency.png", 1400, 900)
    shoot(f"{base}/#/model", DOCS / "model-dark.png", 1400, 1250, dark=True)


if __name__ == "__main__":
    main()
