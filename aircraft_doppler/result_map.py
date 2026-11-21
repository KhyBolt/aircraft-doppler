"""Build a self-contained result map from existing guarded solver artifacts."""

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .wide import STAGES, SURVEY_M, XYZ

WORLD_M = 2 * np.pi * 6378137.


@dataclass
class ScoredGrid:
    spacing_m: int
    rows: np.ndarray
    first_evaluated_spacing_m: np.ndarray
    attempted_positions: int

    def footprint(self, half_width_m):
        """Return the exact union of scored cells as contiguous east-west strips."""
        half_cell = self.spacing_m / 2
        boxes = []
        by_north = sorted(self.rows[:, [1, 0]].tolist())
        for north, group in groupby(by_north, key=lambda point: point[0]):
            east = np.array([point[1] for point in group])
            breaks = np.flatnonzero(np.diff(east) > self.spacing_m) + 1
            for run in np.split(east, breaks):
                boxes.append([max(-half_width_m, run[0] - half_cell),
                              max(-half_width_m, north - half_cell),
                              min(half_width_m, run[-1] + half_cell),
                              min(half_width_m, north + half_cell)])
        return boxes


def read_scored_grids(directory):
    """Read finite stage scores and identify positions already in the solver's cache."""
    grids = []
    first_spacing = {}
    for spacing in (SURVEY_M, *(stage[0] for stage in STAGES)):
        path = directory / f"grid_{spacing}m.csv"
        if not path.is_file():
            continue
        table = pd.read_csv(path, usecols=XYZ + ["loss_hz2"], float_precision="round_trip")
        rows = table[XYZ + ["loss_hz2"]].to_numpy()
        original = np.array([first_spacing.setdefault(tuple(point), spacing)
                             for point in rows[:, :3]], dtype=int)
        finite = np.isfinite(rows[:, 3])
        if finite.any():
            grids.append(ScoredGrid(spacing, rows[finite], original[finite], len(rows)))
    return grids


def write_evaluated_grid(path, grids, result):
    """Save exact ENU centers, training scores, and scored-cell coverage."""
    if not grids:
        return
    half = result["area_width_m"] / 2
    data = {
        "coordinate_frame": "East, north, up metres in the local WGS84 tangent plane",
        "origin_lat_lon_ellipsoid_m": result["area_center_lat_lon_ellipsoid_m"],
        "search_half_width_m": half,
        "point_columns": ["east_m", "north_m", "up_m", "min_training_mse_hz2",
                          "first_evaluated_spacing_m"],
        "footprint_box_columns": ["min_east_m", "min_north_m", "max_east_m", "max_north_m"],
        "stages": [{
            "spacing_m": grid.spacing_m,
            "attempted_positions": grid.attempted_positions,
            "points": [list(row) + [first] for row, first in
                       zip(grid.rows.tolist(), grid.first_evaluated_spacing_m.tolist())],
            "footprint_boxes_en_m": grid.footprint(half),
        } for grid in grids],
    }
    with path.open("x", encoding="utf-8") as out:
        json.dump(data, out, separators=(",", ":"), allow_nan=False)
        out.write("\n")


def _project(local, to_lat_lon, center_lon, mercator):
    points = np.asarray(local)
    if not mercator:
        return points[..., :2] * (1, -1)
    lat, lon = np.moveaxis(np.asarray(to_lat_lon(points)), -1, 0)
    lon = center_lon + (lon - center_lon + 180.) % 360. - 180.
    radians = np.deg2rad(np.clip(lat, -85.05112878, 85.05112878))
    return WORLD_M * np.stack(((lon + 180.)/360., .5 - np.arcsinh(np.tan(radians))/(2*np.pi)), axis=-1)


def _utc(epoch):
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()


def write_result_map(path, result, fit, grids, tracks, to_lat_lon):
    """Keep diagnostic geometry separate from the stable inference JSON schema."""
    latitude, longitude, _ = result["area_center_lat_lon_ellipsoid_m"]
    half = result["area_width_m"] / 2
    mercator = abs(latitude) < 84.

    def project(points):
        return np.round(_project(points, to_lat_lon, longitude, mercator), 3).tolist()

    corners = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]])
    layers = []
    for grid in grids:
        spacing, rows = grid.spacing_m, grid.rows
        cells = rows[:, None, :3] + corners * spacing/2
        cells[:, :, :2] = np.clip(cells[:, :, :2], -half, half)
        mapped_cells = project(cells)
        extent = np.asarray(mapped_cells).reshape(-1, 2)
        layers.append({"spacing_m": spacing, "mse_hz2": rows[:, 3].tolist(),
                       "centers_en_m": rows[:, :2].tolist(),
                       "first_evaluated_spacing_m": grid.first_evaluated_spacing_m.tolist(),
                       "bounds": [extent.min(axis=0).tolist(), extent.max(axis=0).tolist()],
                       "cells": mapped_cells})
    if layers:
        scores = np.concatenate([grid.rows[:, 3] for grid in grids])
        score_range = [float(scores.min()), float(np.quantile(scores, .95))]
    else:
        score_range = None

    accepted = result["status"] == "supported_under_model"
    diagnostic = fit.get("best_fit_diagnostic", {})
    pair = result["estimated_pair"] if accepted else diagnostic.get("pair", "").split("|")
    paths = []
    for identifier in pair:
        if identifier in tracks:
            points, start, end = tracks[identifier]
            paths.append({"segment": identifier, "coordinates": project(points),
                          "start_utc": _utc(start), "end_utc": _utc(end)})

    receiver = None
    if accepted:
        receiver = {"coordinates": project(result["estimated_enu_m"]),
                    "lat_lon": result["estimated_lat_lon"]}
    data = {
        "accepted": accepted,
        "reasons": result["abstention_reasons"],
        "center": [latitude, longitude],
        "clock_anchor": result["supplied_start_utc"],
        "clock_half_width_s": result["clock_half_width_s"],
        "audio_ridges": result["ridges_detected"],
        "catalogue_segments": result["track_segments"],
        "fingerprints": result.get("input_sha256", {}),
        "boundary": project(corners * half),
        "area_width_m": result["area_width_m"],
        "projection": "web_mercator" if mercator else "local",
        "local_m_per_map_m": float(np.cos(np.deg2rad(latitude))) if mercator else 1.,
        "grid": layers,
        "survey_spacing_m": SURVEY_M,
        "score_range": score_range,
        "tracks": paths,
        "receiver": receiver,
        "per_event_training_rmse_hz": diagnostic.get("per_event_rmse_hz"),
        "sampled_near_best_span_m": fit.get("near_best_grid", {}).get("span_xy_m"),
    }
    # Data is embedded in an HTML script element. Escape delimiters in caller-supplied IDs.
    payload = json.dumps(data, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("result_map.html").read_text(encoding="utf-8")
    with path.open("x", encoding="utf-8") as out:
        out.write(template.replace("__RESULT_DATA__", payload, 1))
