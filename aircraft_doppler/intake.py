"""Adapt local media and normalized WGS84 tracks to the guarded solver."""

from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from .io import load_tracks
from .physics import geodetic_to_ecef
from .result_map import diagnostic_candidate, read_scored_grids, write_evaluated_grid, write_result_map
from .wide import BudgetExceeded, CLOCK_HALF_S, EDGE_M, SURVEY_M, XYZ, _sha256, infer


def _utc(value):
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("--start-utc must be an ISO 8601 timestamp with a timezone, such as 2024-04-07T16:34:18Z") from error
    if stamp.tzinfo is None:
        raise ValueError("--start-utc must include a timezone (for example Z or +00:00)")
    return stamp.astimezone(timezone.utc)


def _local_frame(area_center):
    lat, lon, height = area_center
    origin = geodetic_to_ecef(lat, lon, height)
    a, b = np.deg2rad([lat, lon])
    enu = np.array([[-np.sin(b), np.cos(b), 0],
                    [-np.sin(a)*np.cos(b), -np.sin(a)*np.sin(b), np.cos(a)],
                    [np.cos(a)*np.cos(b), np.cos(a)*np.sin(b), np.sin(a)]])
    return origin, enu


def _lat_lon(point):
    """Convert a near-surface ECEF point to WGS84 latitude and longitude."""
    x, y, z = np.moveaxis(np.asarray(point), -1, 0)
    p = np.hypot(x, y)
    semi_major = 6378137.
    semi_minor = 6356752.314245
    e2 = 1 - (semi_minor / semi_major)**2
    param = np.arctan2(z * semi_major, p * semi_minor)
    lat = np.arctan2(z + e2*semi_major**2/semi_minor*np.sin(param)**3,
                     p - e2*semi_major*np.cos(param)**3)
    return np.stack((np.rad2deg(lat), np.rad2deg(np.arctan2(y, x))), axis=-1).tolist()


def locate_recording(audio, tracks, start_utc, area_center, output, *, channel=0, max_seconds=120, published_reference=None):
    """Run the guarded two-pass profile with a local flat-height prior."""
    start = perf_counter()
    anchor = _utc(start_utc)
    origin, enu = _local_frame(area_center)
    if published_reference is not None:
        # Validate before running; this point is never used by the solver.
        geodetic_to_ecef(*published_reference, area_center[2])

    def to_lat_lon(points):
        return _lat_lon(np.asarray(points) @ enu + origin)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory")

    catalogue, frame = load_tracks(tracks, return_frame=True)
    if frame != "ecef":
        raise ValueError("--tracks must supply WGS84 latitude, longitude, and altitude_ellipsoid_m")
    # Retain every supplied segment and its aircraft identity for association.
    local_tracks = []
    track_geometry = {}
    for track in catalogue:
        points = (track.xyz - origin) @ enu.T
        track_geometry[track.flight_id] = (points, track.t[0], track.t[-1])
        table = pd.DataFrame(points, columns=XYZ)
        table["time_s"] = track.t - anchor.timestamp()
        table["flight_id"] = track.flight_id
        table["aircraft_id"] = track.aircraft_id
        table["engine"] = track.engine
        local_tracks.append(table)

    with TemporaryDirectory(prefix="aircraft-doppler-") as scratch:
        scratch = Path(scratch)
        track_csv = scratch / "tracks.csv"
        grid_csv = scratch / "candidates.csv"
        pd.concat(local_tracks).to_csv(track_csv, index=False)
        lattice = range(-EDGE_M, EDGE_M + 1, SURVEY_M)
        pd.DataFrame(product(lattice, lattice, (0,)), columns=XYZ).to_csv(grid_csv, index=False)
        guarded = scratch / "guarded"
        try:
            fit = infer(audio, track_csv, grid_csv, guarded, hardened=True,
                        channel=channel, deadline=start + max_seconds)
        except BudgetExceeded as error:
            partial = guarded / "inference.json"
            fit = (json.loads(partial.read_text(encoding="utf-8")) if partial.is_file()
                   else {"status": "timed_out", "abstention_reasons": [str(error)]})
        grids = read_scored_grids(guarded)

    ridges = fit.get("audio_only_detection", fit.get("audio_audit", {}).get("accepted", []))
    result = {
        "status": "timed_out" if fit["status"] == "timed_out" else "abstained",
        "supplied_start_utc": anchor.isoformat(),
        "clock_half_width_s": CLOCK_HALF_S,
        "area_center_lat_lon_ellipsoid_m": list(area_center),
        "area_width_m": 2 * EDGE_M,
        "channel": channel,
        "recording_duration_s": fit.get("recording_duration_s"),
        "ridges_detected": None if fit["status"] == "timed_out" and "audio_audit" not in fit else len(ridges),
        "track_segments": len(catalogue),
        "abstention_reasons": fit.get("abstention_reasons", []),
    }
    if "inputs_sha256" in fit:
        result["input_sha256"] = {"audio": fit["inputs_sha256"]["wav"], "tracks": _sha256(Path(tracks))}
    if fit["status"] == "supported_in_this_fixture":
        point = np.asarray(fit["estimated_xyz_m"])
        result.update({
            "status": "supported_under_model",
            "evidence_grade": fit["evidence_grade"],
            "estimated_lat_lon": _lat_lon(point @ enu + origin),
            "estimated_enu_m": point.tolist(),
            "estimated_pair": fit["estimated_pair"].split("|"),
            "estimated_audio_start_utc": datetime.fromtimestamp(
                anchor.timestamp() + fit["estimated_clock_offset_s"], timezone.utc).isoformat(),
        })
    result["diagnostic_candidate"] = diagnostic_candidate(
        fit.get("best_fit_diagnostic"), to_lat_lon, result["status"])
    with (output / "inference.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    write_evaluated_grid(output / "evaluated-grid.json", grids, result)
    write_result_map(output / "result-map.html", result, fit, grids, track_geometry, to_lat_lon,
                     published_reference=published_reference)
    return result
