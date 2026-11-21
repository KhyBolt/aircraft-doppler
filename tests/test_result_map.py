import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from aircraft_doppler.result_map import _project, read_scored_grids, write_result_map


def map_data(path):
    html = path.read_text(encoding="utf-8")
    return json.loads(re.search(r'<script type="application/json" id="result-data">(.*?)</script>',
                                html, re.DOTALL)[1])


class ResultMapTests(unittest.TestCase):
    def test_grids_use_only_actual_finite_stage_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "grid_500m.csv").write_text(
                "x_m,y_m,z_m,loss_hz2\n0,0,0,0.125\n500,0,0,\n")
            (path / "grid_25m.csv").write_text(
                "x_m,y_m,z_m,loss_hz2\n25,0,0,0.067\n50,0,0,inf\n")
            grids = read_scored_grids(path)
            self.assertEqual([spacing for spacing, _ in grids], [500, 25])
            self.assertEqual([rows.tolist() for _, rows in grids],
                             [[[0., 0., 0., .125]], [[25., 0., 0., .067]]])

    def test_map_tracks_are_selected_and_abstention_never_serializes_a_pin(self):
        segment = "sim-</script><img src=x>"
        tracks = {name: (np.array([[-1000, 0, 400], [1000, 400, 401]]), 100., 112.)
                  for name in (segment, "sim-2", "decoy")}
        base = {
            "area_center_lat_lon_ellipsoid_m": [88, 7, 2],
            "area_width_m": 10000,
            "supplied_start_utc": "2024-04-07T16:34:18+00:00",
            "clock_half_width_s": 80,
            "ridges_detected": 2,
            "track_segments": 3,
            "estimated_pair": [segment, "sim-2"],
            "estimated_enu_m": [975, -1550, 0],
            "estimated_lat_lon": [88.1, 7.1],
        }
        fit = {"best_fit_diagnostic": {"pair": f"{segment}|sim-2", "xyz_m": [975, -1550, 0]}}
        grid = [(500, np.array([[5000, 0, 0, 0.125]]))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for status in ("supported_under_model", "abstained"):
                with self.subTest(status=status):
                    out = path / (status + ".html")
                    reasons = [] if status == "supported_under_model" else ["Failed gate"]
                    write_result_map(out, dict(base, status=status, abstention_reasons=reasons),
                                     fit, grid, tracks, None)
                    self.assertNotIn(segment, out.read_text(encoding="utf-8"))
                    data = map_data(out)
                    self.assertEqual(data["projection"], "local")
                    self.assertEqual(data["reasons"], reasons)
                    self.assertEqual([track["segment"] for track in data["tracks"]], [segment, "sim-2"])
                    self.assertEqual(data["grid"][0]["mse_hz2"], [0.125])
                    self.assertEqual(max(corner[0] for corner in data["grid"][0]["cells"][0]), 5000.)
                    if status == "supported_under_model":
                        self.assertEqual(data["receiver"]["lat_lon"], [88.1, 7.1])
                    else:
                        self.assertIsNone(data["receiver"])
                        self.assertNotIn("xyz_m", data)

    def test_mercator_wraps_a_regional_track_near_the_dateline(self):
        point = np.array([[0, 179.95], [0, -179.95]])
        cast = _project(point, lambda value: value, 179.9, True)
        self.assertLess(abs(cast[0, 0] - cast[1, 0]), 12000)


if __name__ == "__main__":
    unittest.main()
