"""Integration test: ASSIST ephemeris vs JPL Horizons for Apophis.

Verifies that the :func:`assist.query.integrate` Python API and the ``assist``
CLI entry point work end-to-end, and that the integrated trajectory agrees
with JPL Horizons to within 1 km at every sampled epoch.

Apophis parameters (from the Apophis.ipynb example notebook):
  tstart    = 2.4621385359989386e6 JD
  tstop     = 2.4625030372426095e6 JD
  N_samples = 10000
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Skip the test if the BSP files are not available
# ---------------------------------------------------------------------------
try:
    from _data_paths import data_path
    _PLANETS_BSP   = data_path("de440.bsp")
    _ASTEROIDS_BSP = data_path("sb441-n16.bsp")
    _BSP_AVAILABLE = os.path.exists(_PLANETS_BSP) and os.path.exists(_ASTEROIDS_BSP)
except ImportError:
    _BSP_AVAILABLE = False
    _PLANETS_BSP = _ASTEROIDS_BSP = ""


@unittest.skipUnless(_BSP_AVAILABLE, "BSP ephemeris files not available")
class TestApophisVsHorizons(unittest.TestCase):
    """Compare ASSIST integration of Apophis against JPL Horizons."""

    TSTART   = 2.4621385359989386e6  # JD
    TSTOP    = 2.4625030372426095e6  # JD
    N_SAMPLES = 10000
    AU_TO_KM = 149597870.700          # km per AU

    def setUp(self):
        # Import here so the test is skippable without import errors
        from assist.query import integrate
        from astroquery.jplhorizons import Horizons
        self._integrate = integrate
        self._Horizons  = Horizons

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _fetch_horizons(self, jd_epochs):
        """Return a dict {jd: (x, y, z)} from Horizons (barycentric, ecliptic).

        Queries are chunked to stay within the Horizons API epoch limit.
        """
        CHUNK = 500
        rows_by_jd = {}
        for start in range(0, len(jd_epochs), CHUNK):
            chunk = jd_epochs[start : start + CHUNK]
            obj = self._Horizons(id="Apophis", location="@0", epochs=chunk)
            vecs = obj.vectors(refplane="ecliptic")
            for row in vecs:
                rows_by_jd[float(row["datetime_jd"])] = (
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                )
        return rows_by_jd

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_apophis_vs_horizons(self):
        """ASSIST trajectory must agree with Horizons to within 1 km at every step."""
        tstart    = self.TSTART
        tstop     = self.TSTOP
        n         = self.N_SAMPLES
        tstep     = (tstop - tstart) / (n - 1)

        # --- ASSIST integration ---
        results = self._integrate(
            "Apophis", tstart, tstop, tstep,
            planets_bsp=_PLANETS_BSP,
            asteroids_bsp=_ASTEROIDS_BSP,
        )
        self.assertEqual(len(results), n,
                         f"Expected {n} output steps, got {len(results)}")

        # --- Horizons reference ---
        jd_times   = [sv.t for sv in results]
        horizons   = self._fetch_horizons(jd_times)

        # --- Compute deviations ---
        deviations_km = []
        for sv in results:
            # Find the closest Horizons epoch (floating-point safe)
            jd = min(horizons.keys(), key=lambda j: abs(j - sv.t))
            hx, hy, hz = horizons[jd]
            dx = (sv.x - hx) * self.AU_TO_KM
            dy = (sv.y - hy) * self.AU_TO_KM
            dz = (sv.z - hz) * self.AU_TO_KM
            deviations_km.append(math.sqrt(dx * dx + dy * dy + dz * dz))

        total_abs_dev_km = sum(deviations_km)
        max_dev_km       = max(deviations_km)
        final_dev_km     = deviations_km[-1]

        print(
            f"\nApophis vs Horizons ({n} samples, "
            f"JD {tstart:.4f} – {tstop:.4f}):\n"
            f"  Final deviation:          {final_dev_km * 1e3:10.2f} m\n"
            f"  Max deviation (any step): {max_dev_km  * 1e3:10.2f} m\n"
            f"  Total absolute deviation: {total_abs_dev_km:10.4f} km"
        )

        self.assertLess(
            max_dev_km, 1.0,
            f"Maximum position deviation {max_dev_km:.4f} km ≥ 1 km threshold",
        )


@unittest.skipUnless(_BSP_AVAILABLE, "BSP ephemeris files not available")
class TestAssistCLI(unittest.TestCase):
    """Smoke-test the ``assist`` command-line entry point."""

    TSTART = 2.4621385359989386e6
    TSTOP  = TSTART + 1.0   # 1-day integration (quick smoke test)
    TSTEP  = 0.5

    def _run_cli(self, *extra_args):
        env = os.environ.copy()
        env.setdefault("ASSIST_DIR", os.path.dirname(os.path.dirname(_PLANETS_BSP)))
        cmd = [
            sys.executable, "-m", "assist.query",
            "Apophis",
            str(self.TSTART), str(self.TSTOP), str(self.TSTEP),
        ] + list(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, env=env)

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


@unittest.skipUnless(_BSP_AVAILABLE, "BSP ephemeris files not available")
class TestPythonAPI(unittest.TestCase):
    """Smoke-test the :func:`assist.query.integrate` Python API."""

    def test_integrate_returns_state_vectors(self):
        from assist.query import StateVector, integrate
        tstart = 2.4621385359989386e6
        tstop  = tstart + 2.0
        tstep  = 1.0
        results = integrate(
            "Apophis", tstart, tstop, tstep,
            planets_bsp=_PLANETS_BSP,
            asteroids_bsp=_ASTEROIDS_BSP,
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)  # t=tstart, tstart+1, tstart+2
        for i, sv in enumerate(results):
            self.assertIsInstance(sv, StateVector)
            self.assertAlmostEqual(sv.t, tstart + i * tstep, places=6)

    def test_integrate_raises_on_bad_params(self):
        from assist.query import integrate
        tstart = 2.4621385359989386e6
        with self.assertRaises(ValueError):
            integrate("Apophis", tstart, tstart - 1, 1.0,
                      planets_bsp=_PLANETS_BSP,
                      asteroids_bsp=_ASTEROIDS_BSP)
        with self.assertRaises(ValueError):
            integrate("Apophis", tstart, tstart + 1, -0.5,
                      planets_bsp=_PLANETS_BSP,
                      asteroids_bsp=_ASTEROIDS_BSP)


if __name__ == "__main__":
    unittest.main()
