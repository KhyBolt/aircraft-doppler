"""Build a self-contained result map from existing guarded solver artifacts."""

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .physics import geodetic_to_ecef
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


def diagnostic_candidate(best, to_lat_lon, status):
    """Expose a finite sampled best fit without claiming a supported location."""
    if not best or "xyz_m" not in best or "loss_hz2" not in best:
        return None
    enu = np.asarray(best["xyz_m"], dtype=float)
    mse = float(best["loss_hz2"])
    if enu.shape != (3,) or not np.isfinite(enu).all() or not np.isfinite(mse) or mse < 0:
        return None
    coordinates = np.asarray(to_lat_lon(enu), dtype=float)
    if (coordinates.shape != (2,) or not np.isfinite(coordinates).all()
            or np.any(abs(coordinates) > [90, 180])):
        return None
    label = {"timed_out": "partial_fit_not_accepted",
             "supported_under_model": "also_supported",
             "supported_under_test_profile": "also_supported"}.get(status, "not_accepted")
    return {"status": label, "lat_lon": coordinates.tolist(), "enu_m": enu.tolist(),
            "training_mse_hz2": mse}


def great_circle_m(a, b):
    """Display-only great-circle separation between two WGS84 lat/lon pairs."""
    first, second = np.deg2rad([a, b])
    dlat, dlon = first - second
    h = (np.sin(dlat/2)**2 + np.cos(first[0])*np.cos(second[0])*np.sin(dlon/2)**2)
    return float(6371008.8*2*np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def _reference_location(reference, origin, mercator):
    lat, lon, height = origin
    position = geodetic_to_ecef(*reference, height)
    if mercator:
        xy = _project(reference, lambda point: point, lon, True)
    else:
        a, b = np.deg2rad([lat, lon])
        axes = np.array([[-np.sin(b), np.cos(b), 0],
                        [np.sin(a)*np.cos(b), np.sin(a)*np.sin(b), -np.cos(a)]])
        xy = axes @ (position - geodetic_to_ecef(lat, lon, height))
    return {"coordinates": np.round(xy, 3).tolist(), "lat_lon": list(reference)}


def _utc(epoch):
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()


def write_result_map(path, result, fit, grids, tracks, to_lat_lon, *, published_reference=None):
    """Embed sampled cells and distinct model, diagnostic and published markers."""
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

    accepted = result["status"] in {"supported_under_model", "supported_under_test_profile"}
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
    candidate = result.get("diagnostic_candidate")
    if candidate is not None:
        candidate = {**candidate, "coordinates": project(candidate["enu_m"])}
    ref = (_reference_location(published_reference, result["area_center_lat_lon_ellipsoid_m"], mercator)
           if published_reference is not None else None)
    comparison = None
    if ref is not None and (receiver is not None or candidate is not None):
        comparison = great_circle_m((receiver or candidate)["lat_lon"], ref["lat_lon"])
    data = {
        "accepted": accepted,
        "status": result["status"],
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
        "diagnostic_candidate": candidate,
        "published_reference": ref,
        "reference_separation_m": comparison,
        "per_event_training_rmse_hz": diagnostic.get("per_event_rmse_hz"),
        "sampled_near_best_span_m": fit.get("near_best_grid", {}).get("span_xy_m"),
    }
    # Data is embedded in an HTML script element. Escape delimiters in caller-supplied IDs.
    payload = json.dumps(data, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("result_map.html").read_text(encoding="utf-8")
    with path.open("x", encoding="utf-8") as out:
        out.write(template.replace("__RESULT_DATA__", payload, 1))
