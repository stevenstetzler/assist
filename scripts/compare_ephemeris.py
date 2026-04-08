#!/usr/bin/env python3
"""Diagnostic script: compare ASSIST ephemeris against JPL Horizons for Apophis.

Generates a side-by-side comparison table and a distance-over-time plot.

Usage
-----
::

    # Default: Apophis, full 10,000-sample window, save plot to compare_ephemeris.png
    python scripts/compare_ephemeris.py

    # Custom object / window
    python scripts/compare_ephemeris.py --desig Ceres --tstart 2462138.5 --tstop 2462148.5 --tstep 1.0

    # Skip the plot (useful in headless environments)
    python scripts/compare_ephemeris.py --no-plot

    # Save table to CSV
    python scripts/compare_ephemeris.py --csv compare_ephemeris.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Default parameters (match the Apophis test case in test_query_integration.py)
# ---------------------------------------------------------------------------
DEFAULT_DESIG   = "Apophis"
DEFAULT_TSTART  = 2.4621385359989386e6
DEFAULT_TSTOP   = 2.4625030372426095e6
DEFAULT_N       = 10000
AU_TO_KM        = 149_597_870.700  # km per AU


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fetch_horizons_batch(desig: str, jd_epochs: list[float]) -> list[tuple[float, ...]]:
    """Return ordered (x, y, z) rows from Horizons, chunked to avoid URL-size limits.

    Uses ``refplane='frame'`` (ICRF J2000) to match ASSIST's native coordinate frame.
    """
    from astroquery.jplhorizons import Horizons

    CHUNK = 50
    rows: list[tuple[float, ...]] = []
    for start in range(0, len(jd_epochs), CHUNK):
        chunk = jd_epochs[start: start + CHUNK]
        obj   = Horizons(id=desig, location="@0", epochs=chunk)
        vecs  = obj.vectors(refplane="frame")
        for row in vecs:
            rows.append((float(row["x"]), float(row["y"]), float(row["z"])))
    return rows


def _run_assist(desig: str, tstart: float, tstop: float, tstep: float):
    """Run the ASSIST integration and return a list of StateVector objects."""
    from assist.query import integrate
    return integrate(desig, tstart, tstop, tstep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare ASSIST ephemeris vs JPL Horizons and plot deviation.",
    )
    parser.add_argument("--desig",   default=DEFAULT_DESIG,  help="Object designation")
    parser.add_argument("--tstart",  type=float, default=DEFAULT_TSTART, help="Start JD")
    parser.add_argument("--tstop",   type=float, default=DEFAULT_TSTOP,  help="Stop JD")
    parser.add_argument("--nsamples",type=int,   default=DEFAULT_N,
                        help="Number of evenly-spaced samples (ignored if --tstep is given)")
    parser.add_argument("--tstep",   type=float, default=None,
                        help="Fixed time step (days); overrides --nsamples")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating the distance plot")
    parser.add_argument("--plot-file", default="compare_ephemeris.png",
                        help="Filename for the output plot (default: compare_ephemeris.png)")
    parser.add_argument("--csv",     default=None,
                        help="If given, save the comparison table to this CSV file")
    args = parser.parse_args(argv)

    desig  = args.desig
    tstart = args.tstart
    tstop  = args.tstop

    if args.tstep is not None:
        tstep = args.tstep
    else:
        tstep = (tstop - tstart) / args.nsamples

    print(f"Integrating {desig!r} over JD {tstart:.4f} – {tstop:.4f} (tstep={tstep:.6f} d) …")

    # --- ASSIST ---
    results = _run_assist(desig, tstart, tstop, tstep)
    n = len(results)
    print(f"  ASSIST produced {n} state vectors.")

    # --- Horizons (at the same epochs) ---
    print(f"  Querying Horizons at {n} epochs (batches of 50) …")
    jd_times      = [sv.t for sv in results]
    horizons_rows = _fetch_horizons_batch(desig, jd_times)

    if len(horizons_rows) != n:
        print(f"WARNING: Horizons returned {len(horizons_rows)} rows, expected {n}",
              file=sys.stderr)

    # --- Compute deviations ---
    jds        : list[float] = []
    dists_km   : list[float] = []
    dx_km_list : list[float] = []
    dy_km_list : list[float] = []
    dz_km_list : list[float] = []

    max_dev_km   = 0.0
    max_dev_jd   = 0.0
    total_dev_km = 0.0

    table_rows: list[dict] = []

    for sv, (hx, hy, hz) in zip(results, horizons_rows):
        dx = (sv.x - hx) * AU_TO_KM
        dy = (sv.y - hy) * AU_TO_KM
        dz = (sv.z - hz) * AU_TO_KM
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        jds.append(sv.t)
        dists_km.append(dist)
        dx_km_list.append(dx)
        dy_km_list.append(dy)
        dz_km_list.append(dz)
        total_dev_km += dist

        if dist > max_dev_km:
            max_dev_km = dist
            max_dev_jd = sv.t

        table_rows.append({
            "jd":        sv.t,
            "assist_x":  sv.x,
            "assist_y":  sv.y,
            "assist_z":  sv.z,
            "horizons_x": hx,
            "horizons_y": hy,
            "horizons_z": hz,
            "dx_km":     dx,
            "dy_km":     dy,
            "dz_km":     dz,
            "dist_km":   dist,
        })

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print(f"  Object : {desig}")
    print(f"  Samples: {n}")
    print(f"  Max 3-D deviation  : {max_dev_km:12.4f} km  at JD {max_dev_jd:.6f}")
    print(f"  Final deviation    : {dists_km[-1]:12.4f} km")
    print(f"  Total abs deviation: {total_dev_km:12.4f} km")
    print(f"{'=' * 60}")

    # --- Print first / worst / last rows ---
    col_w = 13
    header = (
        f"{'JD':>18}  {'dx (km)':>{col_w}}  {'dy (km)':>{col_w}}"
        f"  {'dz (km)':>{col_w}}  {'dist (km)':>{col_w}}"
    )
    sep = "-" * len(header)

    def _fmt_row(r: dict) -> str:
        return (
            f"{r['jd']:>18.6f}  {r['dx_km']:>{col_w}.4f}  {r['dy_km']:>{col_w}.4f}"
            f"  {r['dz_km']:>{col_w}.4f}  {r['dist_km']:>{col_w}.4f}"
        )

    worst_idx = dists_km.index(max_dev_km)
    print(f"\nFirst, worst, and last samples:")
    print(header)
    print(sep)
    print(_fmt_row(table_rows[0]))
    if worst_idx not in (0, n - 1):
        print(_fmt_row(table_rows[worst_idx]))
    print(_fmt_row(table_rows[-1]))

    # --- Save CSV ---
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()))
            writer.writeheader()
            writer.writerows(table_rows)
        print(f"\nFull table saved to {args.csv!r}")

    # --- Plot ---
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")   # headless-safe backend
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # top panel: 3-D distance
            ax0 = axes[0]
            ax0.plot(jds, dists_km, color="steelblue", linewidth=0.8, label="3-D distance")
            ax0.axhline(100, color="red",    linestyle="--", linewidth=0.8, label="100 km threshold")
            ax0.axhline(1,   color="orange", linestyle="--", linewidth=0.8, label="1 km threshold")
            ax0.set_ylabel("Distance ASSIST − Horizons (km)")
            ax0.set_title(f"{desig}: ASSIST vs JPL Horizons positional deviation")
            ax0.legend(fontsize=8)
            ax0.grid(True, alpha=0.3)

            # bottom panel: per-component
            ax1 = axes[1]
            ax1.plot(jds, dx_km_list, linewidth=0.6, label="dx")
            ax1.plot(jds, dy_km_list, linewidth=0.6, label="dy")
            ax1.plot(jds, dz_km_list, linewidth=0.6, label="dz")
            ax1.set_xlabel("Julian Date")
            ax1.set_ylabel("Component deviation (km)")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.plot_file, dpi=150)
            plt.close(fig)
            print(f"\nPlot saved to {args.plot_file!r}")
        except ImportError:
            print("\nmatplotlib not installed – skipping plot.", file=sys.stderr)


if __name__ == "__main__":
    main()
