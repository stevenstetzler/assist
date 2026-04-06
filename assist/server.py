"""
FastAPI server for ASSIST ephemeris integration.

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
    uvicorn assist.server:app --reload
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .ephem import Ephem
from .extras import Extras
from .query import StateVector as _StateVector
from .query import integrate as _integrate
from .query import load_ephem, DEFAULT_PLANETS_BSP, DEFAULT_ASTEROIDS_BSP

app = FastAPI(title="ASSIST Ephemeris Server")


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class StateVectorModel(BaseModel):
    t: float    # Julian Date
    x: float    # AU
    y: float    # AU
    z: float    # AU
    vx: float   # AU / day
    vy: float   # AU / day
    vz: float   # AU / day


# ---------------------------------------------------------------------------
# Cache the Ephem object across requests so BSP files are only parsed once.
# ---------------------------------------------------------------------------
_ephem: Ephem | None = None


def get_ephem() -> Ephem:
    global _ephem
    if _ephem is None:
        try:
            _ephem = load_ephem(DEFAULT_PLANETS_BSP, DEFAULT_ASTEROIDS_BSP)
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc))
    return _ephem


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@app.get("/integrate", response_model=List[StateVectorModel])
def integrate_endpoint(
    desig: str = Query(..., description="Object designation (e.g. 'Apophis' or '99942')"),
    tstart: float = Query(..., description="Start Julian Date"),
    tstop: float = Query(..., description="Stop Julian Date"),
    tstep: float = Query(..., description="Output time step in days (> 0)"),
) -> List[StateVectorModel]:
    """Integrate the orbit of *desig* from *tstart* to *tstop* and return
    barycentric ecliptic state vectors sampled every *tstep* days."""

    if tstep <= 0:
        raise HTTPException(status_code=422, detail="tstep must be positive")
    if tstop <= tstart:
        raise HTTPException(status_code=422, detail="tstop must be greater than tstart")

    try:
        ephem = get_ephem()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        results: List[_StateVector] = _integrate(desig, tstart, tstop, tstep, ephem=ephem)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return [StateVectorModel(**asdict(sv)) for sv in results]
