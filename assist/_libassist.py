"""
Internal C library loader and shared helpers.

This module exists to avoid circular imports:
- `assist/__init__.py` re-exports public objects like `Ephem`.
- `assist/ephem.py` needs access to `clibassist` and `assist_error_messages`.

If `assist/ephem.py` imports `assist` (the package top-level) at runtime,
it creates an import cycle. Importing from `assist._libassist` avoids that.
"""

from __future__ import annotations
import os
import site
import sysconfig
import importlib.util
from ctypes import c_char_p, c_int, cdll
from pathlib import Path

def _ext_suffix() -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    return suffix if suffix is not None else ".so"

def _find_libassist() -> str:
    libname = "libassist" + _ext_suffix()
    override = os.environ.get("ASSIST_LIBASSIST_PATH")
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        return str(p)
    candidates = []
    spec = importlib.util.find_spec("assist")
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            candidates.append(Path(loc).parent / libname)
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        candidates.append(Path(sp) / libname)
    candidates.append(Path(__file__).resolve().parent.parent / libname)
    for p in candidates:
        if p.exists():
            return str(p)
    raise OSError(f"Cannot find {libname}. Searched:\n" + "\n".join(f"  {c}" for c in candidates) + "\nSet ASSIST_LIBASSIST_PATH to point at the compiled library.")

clibassist = cdll.LoadLibrary(_find_libassist())
__libpath__ = str(getattr(clibassist, "_name", ""))

def _libassist_str(name: str) -> str:
    val = c_char_p.in_dll(clibassist, name).value
    if val is None:
        raise RuntimeError(f"unable to find symbol {name} in libassist")
    return val.decode("ascii")

__version__ = _libassist_str("assist_version_str")
__build__ = _libassist_str("assist_build_str")
__githash__ = _libassist_str("assist_githash_str")

def assist_error_messages(e: int) -> str:
    e_N = c_int.in_dll(clibassist, "assist_error_messages_N").value
    if e >= e_N:
        raise RuntimeError("An error occured while trying to process an ASSIST error message.")
    ccpp = c_char_p * e_N
    message = ccpp.in_dll(clibassist, "assist_error_messages")[e]
    if message is None:
        raise RuntimeError("assist_error_messages is missing from libassist. Please report this issue on GitHub.")
    return message.decode("ascii")
