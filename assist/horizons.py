"""
Query JPL Horizons for barycentric state vectors and print them in the same
fixed-point format as the ``assist`` CLI.

Entry point: ``horizons desig tstart tstop tstep``
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal


def _fetch_horizons_batch(desig: str, jd_epochs: list, center="@0"):
    """Yield ``(jd, x, y, z, vx, vy, vz)`` rows from Horizons as :class:`Decimal`.

    Queries are chunked into groups of 50 to stay within the API URL-size limit.
    Uses ``refplane='frame'`` (ICRF J2000) to match ASSIST's native coordinate frame.
    """
    from astroquery.jplhorizons import Horizons

    CHUNK = 50
    for start in range(0, len(jd_epochs), CHUNK):
        chunk = jd_epochs[start: start + CHUNK]
        obj = Horizons(id=desig, location=center, epochs=chunk)
        vecs = obj.vectors(refplane="frame")
        for row in vecs:
            yield (
                Decimal(row["datetime_jd"]),
                Decimal(row["x"]),
                Decimal(row["y"]),
                Decimal(row["z"]),
                Decimal(row["vx"]),
                Decimal(row["vy"]),
                Decimal(row["vz"]),
            )


def main(argv=None):
    """CLI entry point: ``horizons desig tstart tstop tstep``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="horizons",
        description=(
            "Query JPL Horizons for barycentric state vectors over a time range "
            "and print them in the same fixed-point format as the ``assist`` CLI."
        ),
    )
    parser.add_argument("desig", help="Object designation (e.g. 'Apophis' or '99942')")
    parser.add_argument("tstart", type=float, help="Start Julian Date")
    parser.add_argument("tstop", type=float, help="Stop Julian Date")
    parser.add_argument("tstep", type=float, help="Output time step in days")
    parser.add_argument(
        "--precision",
        type=int,
        default=8,
        metavar="N",
        help="Number of decimal places in fixed-point output (default: 8).",
    )
    parser.add_argument(
        "--center",
        type=str,
        default="@0",
    )

    args = parser.parse_args(argv)

    jd_epochs = [
        args.tstart + i * args.tstep
        for i in range(int((args.tstop - args.tstart) / args.tstep) + 1)
    ]
    rows = _fetch_horizons_batch(args.desig, jd_epochs, center=args.center)
    p = args.precision
    try:
        for jd, x, y, z, vx, vy, vz in rows:
            print(
                f"{jd:0.{p}f} {x:0.{p}f} {y:0.{p}f} {z:0.{p}f}"
                f" {vx:0.{p}f} {vy:0.{p}f} {vz:0.{p}f}"
            )
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)


if __name__ == "__main__":
    main()
