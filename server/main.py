"""
Simple FastAPI server for ASSIST ephemeris integration.

Endpoints
---------
GET /integrate
    Query parameters:
        desig  – object designation understood by JPL Horizons (e.g. "Apophis", "99942")
        tstart – start Julian Date (float)
        tstop  – stop  Julian Date (float)
        tstep  – output time step in days (float)

    Returns a JSON list of records, one per output epoch:
        [{"t": <JD>, "x": ..., "y": ..., "z": ...,
                      "vx": ..., "vy": ..., "vz": ...}, ...]

Usage
-----
    uvicorn server.main:app --reload
"""

import os
from typing import List

import numpy as np
import rebound
import assist

from astroquery.jplhorizons import Horizons
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Ephemeris file paths – override with the ASSIST_DATA_DIR environment variable
# ---------------------------------------------------------------------------
_DATA_DIR = os.environ.get(
    "ASSIST_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)
_PLANETS_BSP = os.path.join(_DATA_DIR, "de440.bsp")
_ASTEROIDS_BSP = os.path.join(_DATA_DIR, "sb441-n16.bsp")

app = FastAPI(title="ASSIST Ephemeris Server")


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class StateVector(BaseModel):
    t: float   # Julian Date
    x: float   # AU
    y: float   # AU
    z: float   # AU
    vx: float  # AU / day
    vy: float  # AU / day
    vz: float  # AU / day


# ---------------------------------------------------------------------------
# Helper: load (and cache) the Ephem object so we don't re-parse the BSP files
# on every request.
# ---------------------------------------------------------------------------
_ephem: assist.Ephem | None = None


def get_ephem() -> assist.Ephem:
    global _ephem
    if _ephem is None:
        for path in (_PLANETS_BSP, _ASTEROIDS_BSP):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"BSP file not found: {path}. "
                    "Run 'python setup.py download_data' first."
                )
        _ephem = assist.Ephem(_PLANETS_BSP, _ASTEROIDS_BSP)
    return _ephem


# ---------------------------------------------------------------------------
# Helper: query JPL Horizons for barycentric state vectors at a given JD
# ---------------------------------------------------------------------------
def _horizons_state(desig: str, jd: float) -> rebound.Particle:
    """Return a rebound.Particle with the barycentric state of *desig* at *jd*.

    The query uses ``location='@0'`` (solar-system barycenter) so the returned
    vectors are already barycentric — no Sun-offset correction is needed.
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


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@app.get("/integrate", response_model=List[StateVector])
def integrate(
    desig: str = Query(..., description="Object designation (e.g. 'Apophis' or '99942')"),
    tstart: float = Query(..., description="Start Julian Date"),
    tstop: float = Query(..., description="Stop Julian Date"),
    tstep: float = Query(..., description="Output time step in days (> 0)"),
):
    """Integrate the orbit of *desig* from *tstart* to *tstop* and return
    barycentric ecliptic state vectors sampled every *tstep* days."""

    if tstep <= 0:
        raise HTTPException(status_code=422, detail="tstep must be positive")
    if tstop <= tstart:
        raise HTTPException(status_code=422, detail="tstop must be greater than tstart")

    # ------------------------------------------------------------------
    # 1. Load ephemeris
    # ------------------------------------------------------------------
    try:
        ephem = get_ephem()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. Query JPL Horizons for the initial barycentric state
    # ------------------------------------------------------------------
    try:
        initial_state = _horizons_state(desig, tstart)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"JPL Horizons query failed: {exc}",
        )

    # ------------------------------------------------------------------
    # 3. Set up REBOUND simulation + ASSIST extras
    # ------------------------------------------------------------------
    t_initial = tstart - ephem.jd_ref  # JD → days relative to jd_ref
    t_final = tstop - ephem.jd_ref

    sim = rebound.Simulation()
    sim.add(initial_state)
    sim.t = t_initial
    sim.ri_ias15.min_dt = 0.001

    extras = assist.Extras(sim, ephem)
    extras.gr_eih_sources = 11  # GR for Sun and all planets

    # ------------------------------------------------------------------
    # 4. Integrate and collect output at each requested epoch
    # ------------------------------------------------------------------
    times = np.arange(t_initial, t_final + tstep * 0.5, tstep)
    # Clamp the last sample to t_final so we don't overshoot
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
