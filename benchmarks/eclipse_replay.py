"""Replay the guarded ridge scorer on historical Eclipse Soundscapes audio."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import wave

import numpy as np
import pandas as pd

from aircraft_doppler.audio import make_spectrogram, read_audio
from aircraft_doppler.benchmark import STFT_SETTINGS, XYZ, _write_json
from aircraft_doppler.corroboration import check_locked_hypothesis, split_trace
from aircraft_doppler.detection import (
    FIELD_BANDS_HZ, InsufficientEvidence, detect_field_events, detect_two_events_among_distractors,
)
from aircraft_doppler.physics import Track, geodetic_to_ecef
from aircraft_doppler.result_map import diagnostic_candidate, great_circle_m, read_scored_grids, write_result_map
from aircraft_doppler.wide import (
    EDGE_M, MAX_SPAN_M, NEAR_BEST_MSE_HZ2, PAIR_MSE_GAP_HZ2, SOUND_MPS, STAGES,
    SURVEY_M, TRACE_RMSE_LIMIT_HZ, BudgetExceeded, Profile, _best_fit, _deadline_check, _modes, _refine,
    _score_stage, _sha256,
)

LAT, LON, ELLIPSOID_HEIGHT_M = 39.73, -84.20, 190.
ORIGIN = geodetic_to_ecef(LAT, LON, ELLIPSOID_HEIGHT_M)
lat, lon = np.deg2rad([LAT, LON])
ENU = np.array([[-np.sin(lon), np.cos(lon), 0],
                [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])
CLOCK_HALF_S = 300
TRACK_GAP_S = 15
ARMS = {
    "registered": (0, 300, 0),
    "extended": (0, 1800, 0),
    "adjacent": (1800, 900, 1800),
    "wrong-time": (0, 300, 7200),
    "wrong-audio": (1800, 300, 0),
}
APRIL_7_START = datetime.fromisoformat("2024-04-07T16:34:18+00:00").timestamp()


def to_geographic(xyz):
    """Return WGS84 latitude and longitude from the local tangent plane."""
    x, y, z = np.moveaxis(np.asarray(xyz) @ ENU + ORIGIN, -1, 0)
    p = np.hypot(x, y)
    semi_major = 6378137.
    semi_minor = 6356752.314245
    e2 = 1 - (semi_minor / semi_major) ** 2
    angle = np.arctan2(z * semi_major, p * semi_minor)
    lat = np.arctan2(z + e2 * semi_major**2 / semi_minor * np.sin(angle)**3,
                     p - e2 * semi_major * np.cos(angle)**3)
    return np.stack([np.rad2deg(lat), np.rad2deg(np.arctan2(y, x))], axis=-1)


def cut_audio(source, destination, offset_s, duration_s):
    with wave.open(str(source), "rb") as src:
        rate = src.getframerate()
        frames = duration_s * rate
        frame_bytes = src.getsampwidth() * src.getnchannels()
        if (offset_s + duration_s) * rate > src.getnframes():
            raise ValueError("Declared window is outside the staged source recording")
        src.setpos(offset_s * rate)
        with wave.open(str(destination), "wb") as dst:
            dst.setparams(src.getparams())
            dst.setnframes(frames)
            for start in range(0, frames, rate):
                count = min(rate, frames - start)
                samples = src.readframes(count)
                if len(samples) != count * frame_bytes:
                    raise ValueError("Staged WAV data ended before the declared window")
                dst.writeframesraw(samples)


def local_tracks(path, start_utc, duration_s):
    data = pd.read_csv(path, dtype={"aircraft_id": str})
    data["time_s"] = data.utc_s - start_utc
    data = data[data.time_s.between(-CLOCK_HALF_S - 110, duration_s + CLOCK_HALF_S + 15)].copy()
    audit = {"nearby_aircraft": data.aircraft_id.nunique(), "points_in_temporal_crop": len(data),
             "short_segments": 0, "non_subsonic_segments": 0}
    points = geodetic_to_ecef(data.latitude, data.longitude, data.altitude_ellipsoid_m)
    data[XYZ] = (points - ORIGIN) @ ENU.T
    tracks = []
    for aircraft_id, aircraft in data.groupby("aircraft_id", sort=True):
        aircraft = aircraft.sort_values("time_s").drop_duplicates("time_s")
        segment_id = (aircraft.time_s.diff().gt(TRACK_GAP_S) | aircraft.new_leg).cumsum()
        for segment, rows in aircraft.groupby(segment_id):
            if len(rows) < 4:
                audit["short_segments"] += 1
                continue
            track = Track(f"{aircraft_id}-{segment}", rows.time_s.to_numpy(), rows[XYZ].to_numpy(),
                          aircraft_id=aircraft_id, max_gap_s=TRACK_GAP_S)
            t = np.linspace(track.t[0], track.t[-1], int(np.ceil((track.t[-1] - track.t[0]) / .25)) + 1)
            if np.any(np.linalg.norm(track.velocity(t), axis=1) >= SOUND_MPS):
                audit["non_subsonic_segments"] += 1
                continue
            tracks.append(track)
    audit["retained_segments"] = len(tracks)
    return tracks, audit


def _render_map(output, report, *, published_reference=None, tracks=()):
    anchor = report["claimed_start_utc_s"]
    geometry = {track.flight_id: (track.xyz, track.t[0]+anchor, track.t[-1]+anchor)
                for track in tracks}
    best = report.get("diagnostic_candidate")
    if best is None and report.get("diagnostic_point_lat_lon") is not None:
        # Existing public replays store the point and per-event RMSE, but their
        # compact evidence summary may not retain the original ENU row.
        enu = report.get("diagnostic_point_enu_m")
        if enu is None:
            position = geodetic_to_ecef(*report["diagnostic_point_lat_lon"], ELLIPSOID_HEIGHT_M)
            enu = (position - ORIGIN) @ ENU.T
        mse = float(np.mean(np.square(report["ridge_rmse_hz"])))
        best = diagnostic_candidate({"xyz_m": enu, "loss_hz2": mse},
                                    lambda _: report["diagnostic_point_lat_lon"], report["status"])
    view = {
        "status": report["status"], "abstention_reasons": report["reasons"],
        "area_center_lat_lon_ellipsoid_m": report["origin_lat_lon_ellipsoid_m"],
        "area_width_m": report["square_width_m"],
        "supplied_start_utc": datetime.fromtimestamp(anchor, timezone.utc).isoformat(),
        "clock_half_width_s": report["clock_half_s"],
        "ridges_detected": report["ridges_selected"],
        "track_segments": report["track_audit"]["retained_segments"],
        "diagnostic_candidate": best,
        "estimated_pair": report.get("diagnostic_track_segments", []),
        "estimated_lat_lon": report.get("estimated_point_lat_lon"),
        "estimated_enu_m": report.get("diagnostic_point_enu_m"),
        "input_sha256": {"audio": report["input_sha256"]["wav"],
                         "tracks": report["input_sha256"]["regional_adsb"]},
    }
    fit = {"best_fit_diagnostic": {
                "pair": "|".join(report.get("diagnostic_track_segments", [])),
                "per_event_rmse_hz": report.get("ridge_rmse_hz"),
           }, "near_best_grid": {"span_xy_m": report.get("near_best_span_enu_m")}}
    name = "result-map-with-reference.html" if published_reference is not None else "result-map.html"
    write_result_map(output / name, view, fit, read_scored_grids(output), geometry, to_geographic,
                     published_reference=published_reference)


def infer(audio, adsb, arm, output, budget_s, detector, final_spacing_m):
    start = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    offset_s, duration_s, clock_s = ARMS[arm]
    start_utc = APRIL_7_START + clock_s
    wav_path = output / "input.wav"
    cut_audio(audio, wav_path, offset_s, duration_s)
    report = {
        "arm": arm, "status": "abstained", "audio_offset_s": offset_s, "duration_s": duration_s,
        "claimed_start_utc_s": start_utc, "clock_half_s": CLOCK_HALF_S,
        "input_sha256": {"wav": _sha256(wav_path), "regional_adsb": _sha256(adsb)},
        "origin_lat_lon_ellipsoid_m": [LAT, LON, ELLIPSOID_HEIGHT_M],
        "square_width_m": EDGE_M * 2, "field_bands_hz": FIELD_BANDS_HZ, "budget_s": budget_s,
        "detector_profile": detector, "final_spacing_m": final_spacing_m,
    }

    tracks = []
    best_fit = None

    def finish(*reasons):
        report["reasons"] = reasons
        report["wall_s"] = round(perf_counter() - start, 3)
        candidate = diagnostic_candidate(best_fit, to_geographic, report["status"])
        report["diagnostic_candidate"] = candidate
        if candidate is not None:
            report.update(diagnostic_point_lat_lon=candidate["lat_lon"],
                          diagnostic_point_enu_m=candidate["enu_m"],
                          diagnostic_track_segments=best_fit["pair"].split("|"),
                          diagnostic_clock_offset_s=best_fit["clock_offset_s"],
                          ridge_rmse_hz=best_fit["per_event_rmse_hz"])
        _write_json(output / "inference.json", report)
        _render_map(output, report, tracks=tracks)
        return report

    sr, samples = read_audio(wav_path)
    spec = make_spectrogram(samples, sr, **STFT_SETTINGS)
    del samples
    tracks, report["track_audit"] = local_tracks(adsb, start_utc, duration_s)
    try:
        _, _, legacy_audit = detect_two_events_among_distractors(spec)
        report["v04_detector"] = {"status": "two_ridges", "audit": legacy_audit}
    except InsufficientEvidence as error:
        report["v04_detector"] = {"status": "abstained", "reason": str(error), "audit": getattr(error, "audit", {})}
    events, report["field_detector"] = detect_field_events(spec, edge_aware=detector == "edge-aware")
    hz_bin = float(np.median(np.diff(spec.f)))
    report["ridges_selected"] = len(events)
    _write_json(output / "ridges.json", [asdict(event) for event in events])
    if not events:
        return finish("No supported descending ridge in the fixed field bands")
    if not tracks:
        return finish("No usable track segments in the declared time window")

    train, held = zip(*(split_trace(event.trace_points) for event in events))
    if any(len(trace) < 4 for trace in train):
        return finish("Too few fitting frames after the timing guard")
    points = [(x, y, 0) for x in range(-EDGE_M, EDGE_M + 1, SURVEY_M)
              for y in range(-EDGE_M, EDGE_M + 1, SURVEY_M)]
    deadline = start + budget_s
    scorer = None
    try:
        scorer = Profile(tracks, events, deadline, trace_override=train, clock_half_s=CLOCK_HALF_S)
        table = _score_stage(scorer, points, output, SURVEY_M)
        report["search_audit"] = dict(scorer.audit)
        if table.empty:
            return finish("No track hypothesis covered every selected ridge under the clock prior")
        tested = set(points)
        step = SURVEY_M
        for next_step, radius, mode_cap, margin in STAGES:
            if next_step < final_spacing_m:
                break
            _deadline_check(deadline)
            lattice = {point for point in tested if all(value % step == 0 for value in point)}
            centers = _modes(table, lattice, step, mode_cap, radius, margin)
            points = _refine(centers, next_step, radius)
            _score_stage(scorer, points, output, next_step)
            tested.update(points)
            table = scorer.table(sorted(tested))
            step = next_step
    except BudgetExceeded as error:
        report["status"] = "timed_out"
        if scorer is not None:
            report["search_audit"] = dict(scorer.audit)
            best_fit = _best_fit(scorer.cached_table(), len(events))
        return finish(str(error))

    table.to_csv(output / "scores.csv", index=False)
    by_pair = table.drop_duplicates("flight_pair")
    by_pair.to_csv(output / "best_per_pair.csv", index=False)
    winner = table.iloc[0]
    best_fit = _best_fit(table, len(events))
    distances = np.sqrt([winner[f"event_{i+1}_loss_hz2"] for i in range(len(events))])
    near = table[table.loss_hz2 <= winner.loss_hz2 + NEAR_BEST_MSE_HZ2].drop_duplicates(XYZ)
    span = np.ptp(near[["x_m", "y_m"]].to_numpy(), axis=0)
    pair_gap = float(by_pair.iloc[1].loss_hz2 - winner.loss_hz2) if len(by_pair) > 1 else None
    within_fine = set(points)
    complete = all((x + dx, y + dy, z) in within_fine
                   for x, y, z in near[XYZ].itertuples(index=False, name=None)
                   for dx, dy in ((0, 0), (step, 0), (-step, 0), (0, step), (0, -step)))
    blocks, reasons = check_locked_hypothesis(winner, tracks, train, held, SOUND_MPS, hz_bin)
    report.update({
        "search_audit": dict(scorer.audit), "hypotheses_ranked": len(by_pair),
        "ridge_rmse_hz": distances.tolist(), "hypothesis_gap_hz2": pair_gap,
        "near_best_count": len(near), "near_best_span_enu_m": span.tolist(),
        "near_best_complete_fine_neighborhood": complete, "corroboration": blocks,
    })
    if len(events) < 2:
        reasons.append("One ridge cannot satisfy the two-independent-pass acceptance gate")
    if final_spacing_m != STAGES[-1][0]:
        reasons.append("The coarser diagnostic profile omits the required 25 m verification grid")
    if max(distances) > TRACE_RMSE_LIMIT_HZ:
        reasons.append("At least one ridge exceeds the absolute training RMSE gate")
    if np.linalg.norm(span) > MAX_SPAN_M or not complete:
        reasons.append("Near-best support is spatially broad or lacks fine neighborhoods")
    if (abs(near[["x_m", "y_m"]]) == EDGE_M).any().any():
        reasons.append("Near-best support touches the search boundary")
    if abs(winner.clock_offset_s) >= CLOCK_HALF_S:
        reasons.append("Best clock offset reaches the global bound")
    if pair_gap is not None and pair_gap < PAIR_MSE_GAP_HZ2:
        reasons.append("Near-tied track hypotheses")
    if not reasons:
        report["status"] = "supported_under_test_profile"
        report["estimated_point_lat_lon"] = to_geographic(best_fit["xyz_m"]).tolist()
    return finish(*reasons)


def evaluate(output, reference):
    """Apply the independently published station coordinates after inference."""
    import json

    report = json.loads((output / "inference.json").read_text(encoding="utf-8"))
    geodetic_to_ecef(*reference, ELLIPSOID_HEIGHT_M)
    result = {"reference_lat_lon": reference, "arm": report["arm"], "status": report["status"]}
    for field in ("diagnostic", "estimated"):
        point = report.get(f"{field}_point_lat_lon")
        error = great_circle_m(reference, point) if point is not None else None
        result[f"{field}_great_circle_error_m"] = error
    accepted = result["estimated_great_circle_error_m"]
    result["hit_within_1000_m"] = accepted is not None and accepted <= 1000
    name = output / "result-map-with-reference.html"
    if name.exists():
        raise ValueError("Remove the existing evaluated map or choose an unevaluated run directory")
    _render_map(output, report, published_reference=reference)
    _write_json(output / "evaluation.json", result)
    return result


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("infer")
    run.add_argument("--audio", type=Path, required=True)
    run.add_argument("--adsb", type=Path, required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--detector", choices=("interior-only", "edge-aware"), default="edge-aware")
    run.add_argument("--final-spacing-m", type=int, choices=(25, 100), default=25)
    run.add_argument("--budget-s", type=int, choices=range(1, 181), default=120, metavar="1..180")
    run.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("evaluate")
    score.add_argument("--reference", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "infer":
        report = infer(args.audio, args.adsb, args.arm, args.output, args.budget_s,
                       args.detector, args.final_spacing_m)
        result = {key: report.get(key) for key in ("arm", "status", "ridges_selected", "diagnostic_candidate", "estimated_point_lat_lon", "reasons", "wall_s")}
    else:
        result = evaluate(args.output, args.reference)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
