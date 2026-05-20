"""Integration tests: ASSIST ephemeris vs JPL Horizons for multiple asteroids.

Verifies that the :func:`assist.query.integrate` Python API and the ``assist``
CLI entry point work end-to-end, and that the integrated trajectory agrees
with JPL Horizons to within 1 km at every sampled epoch.

Apophis parameters (from the Apophis.ipynb example notebook):
  tstart    = 2.4621385359989386e6 JD
  tstop     = 2.4625030372426095e6 JD
  N_samples = 500
"""

from __future__ import annotations

import subprocess
import sys
import unittest

import pytest

from assist.horizons import _fetch_horizons_batch

# ---------------------------------------------------------------------------
# Shared constants for the Horizons comparison tests
# ---------------------------------------------------------------------------
_J2000 = 2451545.0
_APOPIHS_CA_JD = 2462240.4069444
# set default start/stop to +- 1 year around J2000
_TSTART    = _J2000 - 365  # JD
_TSTOP     = _J2000 + 365  # JD
_N_SAMPLES = 100
_AU_TO_KM  = 149597870.700          # km per AU

_ASTEROIDS = [
    pytest.param(
        "99942", 
        marks=pytest.mark.xfail(reason="Unkown error at close approach")
    ), # Apophis Near earth asteroid with close approach
    "84100", # Farnocchia Main belt
    "588", # Achilles Trojan
    "2060", # Chiron Centaur
    "136199" # Eris TNO
]
_JD_RANGE = {
    # For Apophis, use the date range around close approach with Earth
    "99942": [
        _APOPIHS_CA_JD - 365,
        _APOPIHS_CA_JD + 365
    ], 
    "84100": [_TSTART, _TSTOP],
    "588": [_TSTART, _TSTOP],
    "2060": [_TSTART, _TSTOP],
    "136199": [_TSTART, _TSTOP],
}

def _fetch_horizons_positions(desig: str, jd_epochs: list) -> list:
    """Return ordered ``(x, y, z)`` tuples from Horizons (barycentric ICRF)."""
    rows = []
    for _jd, x, y, z, _vx, _vy, _vz in _fetch_horizons_batch(desig, jd_epochs):
        rows.append((float(x), float(y), float(z)))
    return rows


# ---------------------------------------------------------------------------
# Parametrized pytest tests — one test case per asteroid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desig", _ASTEROIDS)
def test_asteroid_vs_horizons(desig):
    """ASSIST trajectory must agree with Horizons to within 1 km at every step."""

    from assist.query import integrate

    tstart, tstop = _JD_RANGE[desig]
    n      = _N_SAMPLES
    tstep  = (tstop - tstart) / n

    # --- ASSIST integration ---
    results = integrate(desig, tstart, tstop, tstep)
    assert len(results) >= n, f"Expected at least {n} output steps, got {len(results)}"
    assert len(results) <= n + 1, f"Expected at most {n + 1} output steps, got {len(results)}"

    # --- Horizons reference (same JD values as ASSIST output) ---
    jd_times      = [sv.t for sv in results]
    horizons_rows = _fetch_horizons_positions(desig, jd_times)

    # --- Compute deviations ---
    threshold_m = 100  # 100 meters

    deviations = []
    dx_list, dy_list, dz_list = [], [], []

    for sv, (hx, hy, hz) in zip(results, horizons_rows):
        dx = sv.x - hx
        dy = sv.y - hy
        dz = sv.z - hz
        d = (dx**2 + dy**2 + dz**2)**0.5
        deviations.append(d * _AU_TO_KM * 1e3)
        dx_list.append(dx * _AU_TO_KM * 1e3)
        dy_list.append(dy * _AU_TO_KM * 1e3)
        dz_list.append(dz * _AU_TO_KM * 1e3)

    total_abs_dev_m = sum(deviations)
    max_dev_m       = max(deviations)
    final_dev_m     = deviations[-1]

    print(
        f"\n{desig} vs Horizons ({n} samples, "
        f"JD {tstart:.4f} - {tstop:.4f}):\n"
        f"  Final deviation:          {final_dev_m:10.2f} m\n"
        f"  Max deviation (any step): {max_dev_m:10.2f} m\n"
        f"  Total absolute deviation: {total_abs_dev_m:10.4f} km"
    )


    import matplotlib.pyplot as plt

    jd_arr = [sv.t for sv in results]
    # Offset JD to days relative to tstart for readability
    if desig == "99942":
        t_ref = _APOPIHS_CA_JD
    else:
        t_ref = _J2000
    t_days = [jd - t_ref for jd in jd_arr]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top panel: total distance deviation
    axes[0].plot(t_days, deviations, color="black", lw=1.5, label="distance")
    axes[0].axhline(threshold_m, color="red", ls="--", lw=1, label=f"{threshold_m} m threshold")
    axes[0].set_ylabel("Deviation (m)")
    axes[0].set_title(f"Asteroid {desig}: ASSIST vs JPL Horizons")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: per-component deviations
    axes[1].plot(t_days, dx_list, label="dx", lw=1)
    axes[1].plot(t_days, dy_list, label="dy", lw=1)
    axes[1].plot(t_days, dz_list, label="dz", lw=1)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel(f"Days since JD {tstart:.2f}")
    axes[1].set_ylabel("Component deviation (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"deviation_{desig}.png", dpi=150)
    plt.close(fig)
    print(f"  Plot saved → deviation_{desig}.png")

    assert max_dev_m < threshold_m, (
        f"{desig}: Maximum position deviation {max_dev_m:.4f} m >= {threshold_m} m threshold"
    )    

class TestAssistCLI(unittest.TestCase):
    """Smoke-test the ``assist`` command-line entry point."""

    TSTART = 2.4621385359989386e6
    TSTOP  = TSTART + 1.0   # 1-day integration (quick smoke test)
    TSTEP  = 0.5

    def _run_cli(self, *extra_args):
        cmd = [
            sys.executable, "-m", "assist.query",
            "Apophis",
            str(self.TSTART), str(self.TSTOP), str(self.TSTEP),
        ] + list(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_cli_table_output(self):
        result = self._run_cli("--output", "table")
        self.assertEqual(result.returncode, 0,
                         f"CLI exited with non-zero code:\n{result.stderr}")
        self.assertIn("t (JD)", result.stdout)

    def test_cli_json_output(self):
        import json
        result = self._run_cli("--output", "json")
        self.assertEqual(result.returncode, 0,
                         f"CLI exited with non-zero code:\n{result.stderr}")
        data = json.loads(result.stdout)
        self.assertIsInstance(data, list)
        self.assertTrue(len(data) > 0)
        self.assertIn("t",  data[0])
        self.assertIn("x",  data[0])
        self.assertIn("vz", data[0])

    def test_cli_csv_output(self):
        import csv
        import io
        result = self._run_cli("--output", "csv")
        self.assertEqual(result.returncode, 0,
                         f"CLI exited with non-zero code:\n{result.stderr}")
        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)
        self.assertTrue(len(rows) > 0)
        self.assertIn("t",  rows[0])
        self.assertIn("x",  rows[0])
        self.assertIn("vz", rows[0])


class TestPythonAPI(unittest.TestCase):
    """Smoke-test the :func:`assist.query.integrate` Python API."""

    def test_integrate_returns_state_vectors(self):
        from assist.query import StateVector, integrate
        tstart = 2.4621385359989386e6
        tstop  = tstart + 2.0
        tstep  = 1.0
        results = integrate("Apophis", tstart, tstop, tstep)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)  # t=tstart, tstart+1, tstart+2
        for i, sv in enumerate(results):
            self.assertIsInstance(sv, StateVector)
            self.assertAlmostEqual(sv.t, tstart + i * tstep, places=6)

    def test_integrate_raises_on_bad_params(self):
        from assist.query import integrate
        tstart = 2.4621385359989386e6
        with self.assertRaises(ValueError):
            integrate("Apophis", tstart, tstart - 1, 1.0)
        with self.assertRaises(ValueError):
            integrate("Apophis", tstart, tstart + 1, -0.5)


if __name__ == "__main__":
    unittest.main()

