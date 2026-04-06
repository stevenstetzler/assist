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
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import rebound
import assist

from astroquery.jplhorizons import Horizons
from astroquery.jplsbdb import SBDB

# ---------------------------------------------------------------------------
# Ephemeris file paths
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = os.path.join(
    os.environ.get("ASSIST_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)
DEFAULT_PLANETS_BSP = os.path.join(DEFAULT_DATA_DIR, "de440.bsp")
DEFAULT_ASTEROIDS_BSP = os.path.join(DEFAULT_DATA_DIR, "sb441-n16.bsp")

# Internal aliases kept for backwards compatibility
_DATA_DIR = DEFAULT_DATA_DIR
_PLANETS_BSP = DEFAULT_PLANETS_BSP
_ASTEROIDS_BSP = DEFAULT_ASTEROIDS_BSP


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
def load_ephem(planets_bsp: str = DEFAULT_PLANETS_BSP, asteroids_bsp: str = DEFAULT_ASTEROIDS_BSP) -> assist.Ephem:
    """Load the ASSIST ephemeris, raising a descriptive error if BSP files are missing."""
    for path in (planets_bsp, asteroids_bsp):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BSP file not found: {path}. "
                "Run 'python setup.py download_data' first."
            )
    return assist.Ephem(planets_bsp, asteroids_bsp)


# Internal alias kept for backwards compatibility
_load_ephem = load_ephem


def query_horizons_state(desig: str, jd: float) -> rebound.Particle:
    """Return a barycentric ecliptic state vector for *desig* at Julian Date *jd*.

    Uses ``location='@0'`` (solar-system barycenter) so no Sun-offset correction
    is required.
    """
    obj = Horizons(id=desig, location="@0", epochs=jd)
    vecs = obj.vectors(refplane="ecliptic")
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

    nongrav = {mp["name"]: float(mp["value"]) for mp in model_pars if "name" in mp and "value" in mp}

    a1 = nongrav.get("A1", 0.0)
    a2 = nongrav.get("A2", 0.0)
    a3 = nongrav.get("A3", 0.0)
    return a1, a2, a3


# ---------------------------------------------------------------------------
# Main integration function
# ---------------------------------------------------------------------------
def integrate(
    desig: str,
    tstart: float,
    tstop: float,
    tstep: float,
    ephem: Optional[assist.Ephem] = None,
    planets_bsp: str = _PLANETS_BSP,
    asteroids_bsp: str = _ASTEROIDS_BSP,
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
        Pre-loaded :class:`assist.Ephem` instance.  If *None*, one is created
        from *planets_bsp* / *asteroids_bsp*.
    planets_bsp:
        Path to the planets BSP file (de440.bsp).
    asteroids_bsp:
        Path to the asteroids BSP file (sb441-n16.bsp).

    Returns
    -------
    list of StateVector
        Barycentric ecliptic positions (AU) and velocities (AU/day) at each
        sampled epoch, including the Julian Date *t*.
    """
    if tstep <= 0:
        raise ValueError("tstep must be positive")
    if tstop <= tstart:
        raise ValueError("tstop must be greater than tstart")

    if ephem is None:
        ephem = _load_ephem(planets_bsp, asteroids_bsp)

    # 1. Initial barycentric state from JPL Horizons
    initial_state = query_horizons_state(desig, tstart)

    # 2. Non-gravitational parameters from JPL SBDB
    a1, a2, a3 = query_sbdb_nongrav(desig)

    # 3. Build simulation
    t_initial = tstart - ephem.jd_ref  # JD → days relative to jd_ref
    t_final = tstop - ephem.jd_ref

    sim = rebound.Simulation()
    sim.add(initial_state)
    sim.t = t_initial
    sim.ri_ias15.min_dt = 0.001

    extras = assist.Extras(sim, ephem)
    extras.gr_eih_sources = 11  # GR for Sun and all planets
    # Apply non-gravitational parameters: A1 (radial), A2 (transverse), A3 (normal)
    # These model rocket-like accelerations (e.g. cometary outgassing, Yarkovsky effect).
    extras.particle_params = np.array([a1, a2, a3])

    # 4. Integrate and sample
    # Add half a step to the upper bound so np.arange includes t_final when it
    # falls exactly on a grid point (floating-point safe).
    # Clamp any overshoot using a small tolerance (1e-9 days ≈ 0.1 ms).
    times = np.arange(t_initial, t_final + tstep * 0.5, tstep)
    times = times[times <= t_final + 1e-9]

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
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    """Command-line interface: ``assist desig tstart tstop tstep``."""
    parser = argparse.ArgumentParser(
        prog="assist",
        description=(
            "Integrate the orbit of a solar-system object using ASSIST and "
            "print barycentric state vectors as JSON."
        ),
    )
    parser.add_argument("desig", help="Object designation (e.g. 'Apophis' or '99942')")
    parser.add_argument("tstart", type=float, help="Start Julian Date")
    parser.add_argument("tstop", type=float, help="Stop Julian Date")
    parser.add_argument("tstep", type=float, help="Output time step in days")
    args = parser.parse_args(argv)

    try:
        results = integrate(args.desig, args.tstart, args.tstop, args.tstep)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps([asdict(sv) for sv in results], indent=2))


if __name__ == "__main__":
    main()
