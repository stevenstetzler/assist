"""Platform-aware data directory helpers and BSP download utilities for ASSIST."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

_FILES = {
    "linux_p1550p2650.440": "https://ssd.jpl.nasa.gov/ftp/eph/planets/Linux/de440/linux_p1550p2650.440",
    "de440.bsp": "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp",
    "sb441-n16.bsp": "https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/sb441-n16.bsp",
}


def get_assist_dir() -> Path:
    """Return the platform-appropriate ASSIST home directory.

    Resolution order:

    1. ``ASSIST_DIR`` environment variable (explicit override).
    2. Linux / BSD – ``$XDG_DATA_HOME/assist`` (default ``~/.local/share/assist``).
    3. Windows  – ``%LOCALAPPDATA%/assist`` (default ``~/AppData/Local/assist``).
    4. macOS and others – ``~/Library/Application Support/assist``.
    """
    if "ASSIST_DIR" in os.environ:
        return Path(os.environ["ASSIST_DIR"])

    if sys.platform.startswith("linux") or "freebsd" in sys.platform:
        xdg_data = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        return Path(xdg_data) / "assist"

    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local_appdata) / "assist"

    # macOS and anything else
    return Path.home() / "Library" / "Application Support" / "assist"

def data_path(p : str) -> str:
    return str(get_assist_dir() / p)

def data_exists() -> bool:
    """Return True if all required BSP files are present in the data directory."""
    data_dir = get_assist_dir()
    return all((data_dir / fname).exists() for fname in _FILES)


def download_files(data_dir: Path | str | None = None) -> None:
    """Download the required files into *data_dir*.

    If *data_dir* is ``None``, files are placed in ``get_data_dir()``.
    Existing files are skipped.
    """
    if data_dir is None:
        data_dir = get_assist_dir()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in _FILES.items():
        dest = data_dir / filename
        if dest.exists():
            print(f"  {filename} already present, skipping.")
            continue
        print(f"  Downloading {filename} from {url} ...")
        urllib.request.urlretrieve(url, str(dest))
        print(f"  Saved to {dest}")


def _check_data() -> None:
    """Check whether required BSP files exist; download them automatically if not."""
    if not data_exists():
        print("ASSIST data files missing. Downloading now...")
        download_files()
