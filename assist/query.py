"""
Core query and integration API for ASSIST.

This module provides:
- :func:`query_horizons_state` – fetch a barycentric state vector from JPL Horizons
- :func:`query_sbdb_nongrav`   – fetch non-gravitational parameters from JPL SBDB
- :func:`integrate`            – run a full ASSIST integration and return state vectors
- :func:`main`                 – CLI entry point (``assist desig tstart tstop tstep``)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import rebound

from astroquery.jplhorizons import Horizons
from astroquery.jplsbdb import SBDB

from .ephem import Ephem, ASSIST_BODY_IDS
from .extras import Extras
from .data import data_path

PLANETS_BSP = data_path("de440.bsp")
ASTEROIDS_BSP = data_path("sb441-n16.bsp")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class StateVector:
    t: float    # Julian Date
    x: float    # AU
    y: float    # AU
    z: float    # AU
    vx: float   # AU / day
    vy: float   # AU / day
    vz: float   # AU / day


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_ephem(planets_bsp: str = PLANETS_BSP, asteroids_bsp: str = ASTEROIDS_BSP) -> Ephem:
    """Load the ASSIST ephemeris, raising a descriptive error if BSP files are missing."""
    for path in (planets_bsp, asteroids_bsp):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BSP file not found: {path}. "
                "Run 'python setup.py download_data' first."
            )
    return Ephem(planets_bsp, asteroids_bsp)


# Internal alias kept for backwards compatibility
_load_ephem = load_ephem


def query_horizons_state(desig: str, jd: float, center="@0") -> rebound.Particle:
    """Return a barycentric ICRF state vector for *desig* at Julian Date *jd*.

    Uses ``location='@0'`` (solar-system barycenter) and ``refplane='frame'``
    (equatorial ICRF J2000), matching the native coordinate frame of the JPL
    binary ephemeris files (de440.bsp / sb441-n16.bsp) used by ASSIST.
    """
    obj = Horizons(id=desig, location=center, epochs=jd)
    vecs = obj.vectors(refplane="frame")
    row = vecs[0]
    return rebound.Particle(
        x=float(row["x"]),
        y=float(row["y"]),
        z=float(row["z"]),
        vx=float(row["vx"]),
        vy=float(row["vy"]),
        vz=float(row["vz"]),
    )


def query_sbdb_nongrav(desig: str) -> tuple[float, float, float]:
    """Query JPL SBDB for non-gravitational parameters A1, A2, A3.

    Returns a ``(A1, A2, A3)`` tuple of floats. Missing or absent parameters
    default to 0.0.
    """
    result = SBDB.query(desig)
    orbit = result.get("orbit", {})
    model_pars = orbit.get("model_pars", []) or []

    return tuple(model_pars[x].value if x in model_pars else 0.0 for x in ["A1", "A2", "A3"])


# ---------------------------------------------------------------------------
# Helpers for ASSIST perturber bodies
# ---------------------------------------------------------------------------
def _find_assist_body_id(desig: str) -> Optional[int]:
    """Return the ASSIST body index if *desig* matches a known perturber, else None.

    Bodies in :data:`ASSIST_BODY_IDS` (planets, Moon, 16 massive asteroids) are
    already embedded in the ephemeris files; their positions must be read directly
    with :meth:`Ephem.get_particle` rather than being re-integrated by ASSIST.
    """
    desig_lower = desig.lower()
    for k, name in ASSIST_BODY_IDS.items():
        if desig_lower == name.lower():
            return k
    return None


# ---------------------------------------------------------------------------
# Main integration function
# ---------------------------------------------------------------------------
def integrate(
    desig: str,
    tstart: float,
    tstop: float,
    tstep: float,
    ephem: Optional[Ephem] = None,
    planets_bsp: str = PLANETS_BSP,
    asteroids_bsp: str = ASTEROIDS_BSP,
) -> List[StateVector]:
    """Integrate the orbit of *desig* from *tstart* to *tstop*.

    Parameters
    ----------
    desig:
        Object designation understood by JPL Horizons (e.g. ``"Apophis"`` or
        ``"99942"``).
    tstart:
        Start Julian Date.
    tstop:
        Stop Julian Date (must be > *tstart*).
    tstep:
        Output cadence in days (must be > 0).
    ephem:
        Pre-loaded :class:`Ephem` instance.  If *None*, one is created
        from *planets_bsp* / *asteroids_bsp*.
    planets_bsp:
        Path to the planets BSP file (de440.bsp).
    asteroids_bsp:
        Path to the asteroids BSP file (sb441-n16.bsp).

    Returns
    -------
    list of StateVector
        Barycentric ICRF positions (AU) and velocities (AU/day) at each
        sampled epoch, including the Julian Date *t*.
    """
    if tstep <= 0:
        raise ValueError("tstep must be positive")
    if tstop <= tstart:
        raise ValueError("tstop must be greater than tstart")

    if ephem is None:
        ephem = _load_ephem(planets_bsp, asteroids_bsp)

    # Add half a step to the upper bound so np.arange includes t_final when it
    # falls exactly on a grid point (floating-point safe).
    # Clamp any overshoot using a small tolerance (1e-9 days ≈ 0.1 ms).
    t_initial = tstart - ephem.jd_ref  # JD → days relative to jd_ref
    t_final   = tstop  - ephem.jd_ref
    times = np.arange(t_initial, t_final + tstep * 0.5, tstep)
    times = times[times <= t_final + 1e-9]

    # -----------------------------------------------------------------------
    # If the target is a known ASSIST perturber body (planet, Moon, or one of
    # the 16 massive asteroids in ASSIST_BODY_IDS), its trajectory is already
    # embedded in the BSP ephemeris files.  Reading it directly with
    # ephem.get_particle() is both faster and more accurate than re-integrating
    # it with ASSIST (which would introduce small discrepancies).
    # -----------------------------------------------------------------------
    body_id = _find_assist_body_id(desig)
    if body_id is not None:
        results: List[StateVector] = []
        for t in times:
            p = ephem.get_particle(body_id, t)
            results.append(
                StateVector(
                    t=t + ephem.jd_ref,
                    x=p.x,
                    y=p.y,
                    z=p.z,
                    vx=p.vx,
                    vy=p.vy,
                    vz=p.vz,
                )
            )
        return results

    # get initial barycentric state from JPL Horizons
    initial_state = query_horizons_state(desig, tstart, center="@0")

    # create rebound simulation with particle at initial state
    sim = rebound.Simulation()
    sim.add(initial_state)
    sim.t = t_initial
    sim.ri_ias15.min_dt = 0.001

    extras = Extras(sim, ephem)
    extras.gr_eih_sources = 11  # GR for Sun and all planets

    # Get non-gravitational parameters from JPL SBDB
    a1, a2, a3 = query_sbdb_nongrav(desig)
    if any([x != 0.0 for x in [a1, a2, a3]]): 
        # add non-grav forces
        extras.particle_params = np.array([a1, a2, a3])

    # 4. Integrate and sample
    results: List[StateVector] = []
    for t in times:
        extras.integrate_or_interpolate(t)
        p = sim.particles[0]
        results.append(
            StateVector(
                t=t + ephem.jd_ref,
                x=p.x,
                y=p.y,
                z=p.z,
                vx=p.vx,
                vy=p.vy,
                vz=p.vz,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------
def _format_ascii(results: List[StateVector], precision: int = 8) -> str:
    """Return a fixed-width ASCII table of state vectors.

    Uses fixed-point (``f``) notation.  *precision* controls the number of
    decimal places; the column width is derived automatically so the header
    labels remain aligned.
    """
    # Column width: room for a sign, up to 3 integer digits, decimal point,
    # and *precision* fractional digits.  Always at least 12 to fit the header.
    w = max(precision + 6, 12)
    header = (
        f"{'t (JD)':>16}  {'x (AU)':>{w}}  {'y (AU)':>{w}}  {'z (AU)':>{w}}"
        f"  {'vx (AU/d)':>{w}}  {'vy (AU/d)':>{w}}  {'vz (AU/d)':>{w}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for sv in results:
        lines.append(
            f"{sv.t:>16.4f}"
            f"  {sv.x:>{w}.{precision}f}  {sv.y:>{w}.{precision}f}  {sv.z:>{w}.{precision}f}"
            f"  {sv.vx:>{w}.{precision}f}  {sv.vy:>{w}.{precision}f}  {sv.vz:>{w}.{precision}f}"
        )
    return "\n".join(lines)


def _format_json(results: List[StateVector]) -> str:
    """Return a JSON array of state vectors."""
    return json.dumps([asdict(sv) for sv in results], indent=2)


def _format_csv(results: List[StateVector]) -> str:
    """Return a CSV-formatted string of state vectors."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["t", "x", "y", "z", "vx", "vy", "vz"])
    for sv in results:
        writer.writerow([sv.t, sv.x, sv.y, sv.z, sv.vx, sv.vy, sv.vz])
    return buf.getvalue().rstrip("\r\n")


_FORMATTERS = {
    "table": _format_ascii,  # called with (results, precision=...)
    "json": _format_json,
    "csv": _format_csv,
}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    """Command-line interface: ``assist desig tstart tstop tstep``."""
    parser = argparse.ArgumentParser(
        prog="assist",
        description=(
            "Integrate the orbit of a solar-system object using ASSIST and "
            "print barycentric state vectors."
        ),
    )
    parser.add_argument("desig", help="Object designation (e.g. 'Apophis' or '99942')")
    parser.add_argument("tstart", type=float, help="Start Julian Date")
    parser.add_argument("tstop", type=float, help="Stop Julian Date")
    parser.add_argument("tstep", type=float, help="Output time step in days")
    parser.add_argument(
        "--output",
        choices=["table", "json", "csv"],
        default="table",
        help=(
            "Output format. 'table' (default) prints a fixed-width ASCII table; "
            "'json' prints machine-readable JSON; 'csv' prints comma-separated values."
        ),
    )
    parser.add_argument(
        "--output-precision",
        type=int,
        default=8,
        metavar="N",
        help=(
            "Number of decimal places in fixed-point output (default: 8). "
            "Applies to the 'table' format."
        ),
    )
    args = parser.parse_args(argv)

    try:
        results = integrate(args.desig, args.tstart, args.tstop, args.tstep)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output == "table":
        output = _format_ascii(results, precision=args.output_precision)
    else:
        output = _FORMATTERS[args.output](results)

    try:
        print(output)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)


if __name__ == "__main__":
    main()
