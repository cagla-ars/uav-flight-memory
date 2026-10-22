from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from uav_flight_memory.config import CSV_ROOT, HF_RESOLVE_BASE


def csv_relative_path(flight_id: str) -> Path:
    return Path("csv_files_annotated") / Path(*flight_id.split("/")).with_suffix(".csv")


def local_csv_path(flight_id: str, csv_root: Path = CSV_ROOT) -> Path:
    return csv_root / Path(*flight_id.split("/")).with_suffix(".csv")


def csv_url(flight_id: str) -> str:
    parts = ["csv_files_annotated", *flight_id.split("/")]
    quoted = "/".join(urllib.parse.quote(part) for part in parts) + ".csv"
    return f"{HF_RESOLVE_BASE}/{quoted}"


def ensure_csv(flight_id: str, csv_root: Path = CSV_ROOT, timeout: int = 60) -> tuple[Path | None, str]:
    path = local_csv_path(flight_id, csv_root)
    if path.exists() and path.stat().st_size > 0:
        return path, "cached"

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(csv_url(flight_id), headers={"User-Agent": "uav-flight-memory/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        return None, f"http_{exc.code}"
    except OSError as exc:
        return None, f"error:{exc.__class__.__name__}"

    return path, "downloaded"

