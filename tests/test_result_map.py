import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from aircraft_doppler.result_map import (
    ScoredGrid, _project, diagnostic_candidate, read_scored_grids, write_evaluated_grid, write_result_map,
)


def map_data(path):
    html = path.read_text(encoding="utf-8")
    return json.loads(re.search(r'<script type="application/json" id="result-data">(.*?)</script>',
                                html, re.DOTALL)[1])


class ResultMapTests(unittest.TestCase):
    def test_grids_keep_real_scores_and_identify_cached_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "grid_500m.csv").write_text(
                "x_m,y_m,z_m,loss_hz2\n0,0,0,0.125\n500,0,0,\n")
            (path / "grid_100m.csv").write_text(
                "x_m,y_m,z_m,loss_hz2\n0,0,0,0.125\n100,0,0,0.11111111111111113\n")
            (path / "grid_25m.csv").write_text(
                "x_m,y_m,z_m,loss_hz2\n0,0,0,0.125\n25,0,0,0.067\n50,0,0,inf\n100,0,0,0.11111111111111113\n")
            grids = read_scored_grids(path)
            self.assertEqual([grid.spacing_m for grid in grids], [500, 100, 25])
            self.assertEqual(grids[-1].attempted_positions, 4)
            self.assertEqual(grids[-1].rows.tolist(), [[0., 0., 0., .125], [25., 0., 0., .067],
                                                     [100., 0., 0., .11111111111111113]])
            self.assertEqual(grids[-1].first_evaluated_spacing_m.tolist(), [500, 25, 100])
            origin = [88, 7, 2]
            out = path / "evaluated-grid.json"
            write_evaluated_grid(out, grids, {"area_width_m": 10000,
                                             "area_center_lat_lon_ellipsoid_m": origin})
            parsed = json.loads(out.read_text())
            self.assertEqual(parsed["origin_lat_lon_ellipsoid_m"], origin)
            self.assertEqual(parsed["stages"][-1]["points"],
                             [[0., 0., 0., .125, 500], [25., 0., 0., .067, 25],
                              [100., 0., 0., .11111111111111113, 100]])
            self.assertEqual(parsed["stages"][-1]["footprint_boxes_en_m"],
                             [[-12.5, -12.5, 37.5, 12.5], [87.5, -12.5, 112.5, 12.5]])

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
        grid = [ScoredGrid(500, np.array([[5000, 0, 0, 0.125]]), np.array([500]), 1)]
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
                    self.assertEqual(data["grid"][0]["centers_en_m"], [[5000, 0]])
                    self.assertEqual(data["grid"][0]["first_evaluated_spacing_m"], [500])
                    self.assertEqual(max(corner[0] for corner in data["grid"][0]["cells"][0]), 5000.)
                    if status == "supported_under_model":
                        self.assertEqual(data["receiver"]["lat_lon"], [88.1, 7.1])
                    else:
                        self.assertIsNone(data["receiver"])
                        self.assertNotIn("xyz_m", data)

    def test_diagnostic_coordinates_require_a_finite_position_and_score(self):
        for best in (None, {}, {"xyz_m": [1,2,3]}, {"xyz_m": [1,2,np.nan], "loss_hz2": 0.},
                     {"xyz_m": [1,2,3], "loss_hz2": np.inf}):
            with self.subTest(best=best):
                self.assertIsNone(diagnostic_candidate(best, lambda _: [0,0], "abstained"))
        self.assertIsNone(diagnostic_candidate({"xyz_m": [0,0,0], "loss_hz2": 0.},
                                              lambda _: [100,0], "abstained"))

    def test_public_eclipse_reference_and_candidate_are_distinct_from_a_fix(self):
        from shutil import copyfile
        from benchmarks.eclipse_replay import evaluate

        evidence = Path(__file__).resolve().parents[1]/"evidence"/"eclipse-505"
        suite = json.loads((evidence/"summary.json").read_text())
        recorded = next(item for item in suite["runs"] if item["run"] == "registered")
        # An already-published field result; the reference is supplied only to evaluation.
        report = {**recorded, "origin_lat_lon_ellipsoid_m": suite["origin_lat_lon_ellipsoid_m"],
                  "square_width_m": suite["grid_width_m"]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/"inference.json").write_text(json.dumps(report))
            copyfile(evidence/"registered-grid-500m.csv", path/"grid_500m.csv")
            result = evaluate(path, recorded["reference_lat_lon"])
            data = map_data(path/"result-map-with-reference.html")
            self.assertEqual(result["status"], "abstained")
            self.assertFalse(result["hit_within_1000_m"])
            self.assertIsNone(data["receiver"])
            self.assertEqual(data["diagnostic_candidate"]["status"], "not_accepted")
            self.assertEqual(data["published_reference"]["lat_lon"], recorded["reference_lat_lon"])
            self.assertAlmostEqual(data["reference_separation_m"], 395, delta=2)
            self.assertEqual([layer["spacing_m"] for layer in data["grid"]], [500])
            self.assertEqual(json.loads((path/"inference.json").read_text()), report)

    def test_mercator_wraps_a_regional_track_near_the_dateline(self):
        point = np.array([[0, 179.95], [0, -179.95]])
        cast = _project(point, lambda value: value, 179.9, True)
        self.assertLess(abs(cast[0, 0] - cast[1, 0]), 12000)


if __name__ == "__main__":
    unittest.main()
