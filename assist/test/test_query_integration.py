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

import math
import os
import subprocess
import sys
import unittest

import numpy as np

from assist.horizons import _fetch_horizons_batch


class TestAsteroidsVsHorizons(unittest.TestCase):
    """Compare ASSIST integrations against JPL Horizons for multiple asteroids."""

    TSTART    = 2.4621385359989386e6  # JD
    TSTOP     = 2.4625030372426095e6  # JD
    N_SAMPLES = 500
    AU_TO_KM  = 149597870.700          # km per AU

    ASTEROIDS = ["Apophis", "Albion", "Achilles", "Chiron", "Eris"]

    def setUp(self):
        from assist.query import integrate
        self._integrate = integrate

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _fetch_horizons(self, desig, jd_epochs):
        """Return an ordered list of (x, y, z) tuples from Horizons (barycentric ICRF).

        Delegates to :func:`assist.horizons._fetch_horizons_batch`, extracting only
        the position components used for comparison.
        """
        rows = []
        for _jd, x, y, z, _vx, _vy, _vz in _fetch_horizons_batch(desig, jd_epochs):
            rows.append((float(x), float(y), float(z)))
        return rows

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_vs_horizons(self):
        """ASSIST trajectory must agree with Horizons to within 1 km at every step."""
        tstart = self.TSTART
        tstop  = self.TSTOP
        n      = self.N_SAMPLES
        tstep  = (tstop - tstart) / n

        for desig in self.ASTEROIDS:
            with self.subTest(desig=desig):
                # --- ASSIST integration ---
                results = self._integrate(desig, tstart, tstop, tstep)
                self.assertGreaterEqual(len(results), n,
                                        f"Expected at least {n} output steps, got {len(results)}")
                self.assertLessEqual(len(results), n + 1,
                                     f"Expected at most {n + 1} output steps, got {len(results)}")

                # --- Horizons reference (same JD values as ASSIST output) ---
                jd_times      = [sv.t for sv in results]
                horizons_rows = self._fetch_horizons(desig, jd_times)

                # --- Compute deviations (AU); assert per-component ---
                AU_TO_KM  = self.AU_TO_KM
                threshold = 100.0 / AU_TO_KM   # 100 km expressed in AU
                deviations = []
                for sv, (hx, hy, hz) in zip(results, horizons_rows):
                    dx = sv.x - hx
                    dy = sv.y - hy
                    dz = sv.z - hz
                    deviations.append(math.sqrt(dx * dx + dy * dy + dz * dz))
                    self.assertLess(
                        abs(dx), threshold,
                        f"dx at JD {sv.t}: {abs(dx) * AU_TO_KM:.2f} km >= 100 km",
                    )
                    self.assertLess(
                        abs(dy), threshold,
                        f"dy at JD {sv.t}: {abs(dy) * AU_TO_KM:.2f} km >= 100 km",
                    )
                    self.assertLess(
                        abs(dz), threshold,
                        f"dz at JD {sv.t}: {abs(dz) * AU_TO_KM:.2f} km >= 100 km",
                    )

                total_abs_dev_km = sum(deviations) * AU_TO_KM
                max_dev_km       = max(deviations) * AU_TO_KM
                final_dev_km     = deviations[-1] * AU_TO_KM

                print(
                    f"\n{desig} vs Horizons ({n} samples, "
                    f"JD {tstart:.4f} - {tstop:.4f}):\n"
                    f"  Final deviation:          {final_dev_km * 1e3:10.2f} m\n"
                    f"  Max deviation (any step): {max_dev_km  * 1e3:10.2f} m\n"
                    f"  Total absolute deviation: {total_abs_dev_km:10.4f} km"
                )

                self.assertLess(
                    max_dev_km, 1.0,
                    f"{desig}: Maximum position deviation {max_dev_km:.4f} km >= 1 km threshold",
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

