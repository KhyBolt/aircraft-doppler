"""Automatic audio ridge extraction and full-day Livermore archive replay."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import maximum_filter1d, percentile_filter
from scipy.optimize import least_squares, minimize_scalar

from aircraft_doppler.audio import make_spectrogram, read_audio
from aircraft_doppler.corroboration import SETTINGS, local_jitter_hz, split_trace
from aircraft_doppler.detection import _ridge
from aircraft_doppler.intake import _lat_lon
from aircraft_doppler.measurement import fit_ideal_trace, ideal_received_frequency
from aircraft_doppler.result_map import great_circle_m
from benchmarks.livermore_archive import branch_tail, catalogue, ecef, fetch, origin, rot, sound
from benchmarks.livermore_inputs import CLIPS, verify


DAY = datetime(2026, 5, 17, tzinfo=timezone.utc).timestamp()
GAP_S = 1027.242
EDGE_M = 5000
CLOCK_STEP_S = 2
EMISSION_STEP_S = 0.25
KNOWN_PAIR = ("ad09eb", "abcd86")
MANIFESTS = {
    "2026-05-17": "d29fb193d0ae54f4f74da4c8186bb11e861d32e4d62332b9abc20a7495d0af47",
    "2026-05-18": "bbdc5046a0456e62f95c3f1bbd3bff52ab31651d7f08ca4facc4050f660c1c5e",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def harmonic_ridge(path):
    rate, audio = read_audio(path)
    spec = make_spectrogram(audio, rate, fmin=40, fmax=800)
    times = spec.t[::2]
    fundamentals = np.arange(70, 191, 0.5)
    power_db = 10 * np.log10(np.maximum(spec.power[:, ::2], 1e-20))
    frequency_contrast = power_db - percentile_filter(power_db, 50, size=(25, 1), mode="nearest")
    temporal_contrast = power_db - np.median(power_db, axis=1, keepdims=True)
    prominence = np.clip(np.minimum(frequency_contrast, temporal_contrast), 0, 26)
    harmonic_indices = np.searchsorted(spec.f, np.arange(1, 5)[:, None] * fundamentals[None, :])
    comb = maximum_filter1d(prominence, 5, axis=0)[harmonic_indices]
    consensus = np.sort(comb, axis=0)
    score = consensus[-2] + consensus[-3] + 0.25 * consensus[-1]
    path_bins = _ridge(score, 0.5, 0.5)
    guided_bins = harmonic_indices[:, path_bins]
    offsets = np.arange(-2, 3)
    choices = guided_bins[:, :, None] + offsets
    cols = np.arange(len(times))
    peak_scores = prominence[choices, cols[None, :, None]]
    select = peak_scores.argmax(axis=2, keepdims=True)
    peaks = np.take_along_axis(choices, select, axis=2)[..., 0]
    peak_scores = np.take_along_axis(peak_scores, select, axis=2)[..., 0]
    left, top, right = [power_db[peaks + delta, cols] for delta in (-1, 0, 1)]
    curvature = left - 2 * top + right
    refined = np.divide(0.5 * (left - right), curvature, out=np.zeros_like(top), where=curvature < -1e-8)
    measured = (spec.f[peaks] + 0.5 * np.clip(refined, -0.5, 0.5)) / np.arange(1, 5)[:, None]
    data = []
    for col, time in enumerate(times):
        supported = (peak_scores[:, col] >= 6) & (top[:, col] >= left[:, col]) & (top[:, col] >= right[:, col])
        votes = measured[supported, col]
        weights = peak_scores[supported, col]
        if len(votes) < 2:
            continue
        agreeing = abs(votes[:, None] - votes[None, :]) <= 0.65
        group = agreeing[np.argmax(agreeing @ weights)]
        if group.sum() >= 2:
            data.append((time, np.average(votes[group], weights=weights[group])))
    trace = np.asarray(data)
    if len(trace) < 50:
        raise ValueError(f"Only {len(trace)} harmonic-consensus frames in {path.name}")
    initial = (80, 2000, np.median(trace[:, 1]), np.median(trace[:, 0]))
    ideal = fit_ideal_trace(*trace.T, initial, sound)
    fit_names = ("speed_mps", "slant_cpa_m", "f0_hz", "delayed_cpa_audio_s")
    expectation = ideal_received_frequency(trace[:, 0], *(ideal[name] for name in fit_names), sound)
    accepted = abs(trace[:, 1] - expectation) <= 3
    trace = trace[accepted]
    if len(trace) < 50:
        raise ValueError(f"Too few coherent frames in {path.name}")
    ideal = fit_ideal_trace(*trace.T, tuple(ideal[name] for name in fit_names), sound)
    frequency_drop = float(np.median(trace[:8, 1]) - np.median(trace[-8:, 1]))
    if frequency_drop < 12 or ideal["rmse_hz"] > 2.75 or not trace[0, 0] + 5 < ideal["delayed_cpa_audio_s"] < trace[-1, 0] - 5:
        raise ValueError(f"Weak or one-sided harmonic-consensus ridge in {path.name}: {ideal}")
    return trace, {"frames": len(trace), "rejected_outlier_frames": int((~accepted).sum()),
                   "frequency_drop_hz": frequency_drop, "ideal_fit": ideal}


def load_shards(directory, day=DAY):
    directory.mkdir(parents=True, exist_ok=True)

    def load(hour):
        path = directory / f"{hour:02}.json"
        if not path.is_file():
            rows, _, _ = fetch((day, hour))
            path.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
        content = path.read_bytes()
        rows = json.loads(content)
        return rows, (hour, len(rows), sha256(content).hexdigest())

    with ThreadPoolExecutor(max_workers=6) as workers:
        paired = list(workers.map(load, range(24)))
    rows, receipt = zip(*paired)
    manifest = sha256(json.dumps(receipt, separators=(",", ":")).encode()).hexdigest()
    return rows, receipt, manifest


def prepare_shard(rows, hour, matching_pair, day=DAY):
    if matching_pair:
        rows = [row for row in rows if row[0] in matching_pair]
    if not any(0 < row[4] < 40000 for row in rows):
        return np.empty(0), []
    catalogue_tracks, identities, width, mid, _ = catalogue(rows, day, hour)
    start = (mid - day) - width
    stop = min(start + 3600, 86400 - branch_tail)
    clocks = np.arange(start, stop, CLOCK_STEP_S)
    flights = []
    for track in catalogue_tracks:
        relative = np.linspace(track.t[0], track.t[-1], 1 + int(np.ceil(np.ptp(track.t) / EMISSION_STEP_S)))
        time = relative + (mid - day)
        position = track.position(relative)
        speed = track.velocity(relative)
        if np.any(np.linalg.norm(speed, axis=1) >= sound):
            continue
        airframe = identities[track.aircraft_id]
        flights.append((airframe, time, position, speed))
    return clocks, flights


def coordinates(point):
    latitude, longitude = _lat_lon(np.asarray(point) @ rot + ecef)
    return [float(longitude), float(latitude)]


def curve(flight, point):
    _, emission, position, speed = flight
    delta = position - point
    distance = np.linalg.norm(delta, axis=1)
    radial = np.einsum("ij,ij->i", delta, speed) / distance
    return emission + distance / sound, 1 / (1 + radial / sound)


def profile(curves, traces, coverage, clocks):
    losses, sources = [], []
    for (arrival, ratio), obs, (first, last) in zip(curves, traces, coverage):
        doppler = np.interp(clocks[:, None] + obs[:, 0], arrival, ratio)
        source = (doppler * obs[:, 1]).sum(axis=1) / (doppler**2).sum(axis=1)
        loss = ((doppler * source[:, None] - obs[:, 1])**2).mean(axis=1)
        loss[(clocks + first < arrival[0]) | (clocks + last > arrival[-1])] = np.inf
        losses.append(loss)
        sources.append(source)
    return np.mean(losses, axis=0), sources


def score_shard(shard, point, traces, coverage, matching_pair):
    clocks, flights = shard
    profiles = [[], []]
    for index, flight in enumerate(flights):
        passage = curve(flight, point)
        for slot, obs in enumerate(traces):
            if matching_pair and flight[0] != matching_pair[slot]:
                continue
            first, last = coverage[slot]
            lo = np.searchsorted(clocks, passage[0][0] - first)
            hi = np.searchsorted(clocks, passage[0][-1] - last, side="right")
            if lo >= hi:
                continue
            losses, _ = profile([passage], [obs], [(first, last)], clocks[lo:hi])
            profiles[slot].append((lo, hi, index, losses))
    if not all(profiles):
        return {}
    second_lo = np.asarray([item[0] for item in profiles[1]])
    second_hi = np.asarray([item[1] for item in profiles[1]])
    minima = {}
    for lo, hi, index, losses in profiles[0]:
        first_id = flights[index][0]
        for candidate_index in np.flatnonzero((second_lo < hi) & (second_hi > lo)):
            other_lo, other_hi, other_index, other_losses = profiles[1][candidate_index]
            second_id = flights[other_index][0]
            if first_id == second_id:
                continue
            lower, upper = max(lo, other_lo), min(hi, other_hi)
            scores = (losses[lower-lo:upper-lo] + other_losses[lower-other_lo:upper-other_lo]) / 2
            best = scores.argmin()
            score = float(scores[best])
            pair = first_id, second_id
            if np.isfinite(score) and (pair not in minima or score < minima[pair][0]):
                minima[pair] = score, float(clocks[lower + best]), index, other_index
    return minima


def coarse_search(shards, traces, coverage, matching_pair):
    scores = {}
    grid = []
    for north in range(-EDGE_M, EDGE_M + 1, 1000):
        for east in range(-EDGE_M, EDGE_M + 1, 1000):
            point = east, north, 0
            score_at_point = np.inf
            for hour, shard in enumerate(shards):
                minima = score_shard(shard, point, traces, coverage, matching_pair)
                for pair, (loss, clock, first, second) in minima.items():
                    if pair not in scores or loss < scores[pair][0]:
                        scores[pair] = loss, clock, point, hour, first, second
                    score_at_point = min(loss, score_at_point)
            grid.append((*point, score_at_point if np.isfinite(score_at_point) else None))
        print(f"coarse northing {north:+} m: {len(scores):,} distinct ordered pairs", flush=True)
    return sorted(scores.items(), key=lambda item: (item[1][0], item[0])), grid


def refine(shards, pair_row, traces, coverage):
    pair, (loss, clock, point, hour, first, second) = pair_row
    flights = [shards[hour][1][index] for index in (first, second)]

    def objective(params, details=False):
        east, north, relative_clock = params
        clip_start = clock + relative_clock
        curves = [curve(flight, (east, north, 0)) for flight in flights]
        values, sources = profile(curves, traces, coverage, np.asarray([clip_start]))
        if details:
            return float(values[0]), [float(source[0]) for source in sources], curves
        residuals = []
        for (arrival, ratio), obs, source in zip(curves, traces, sources):
            prediction = float(source[0]) * np.interp(clip_start + obs[:, 0], arrival, ratio)
            residuals.extend((prediction - obs[:, 1]) / np.sqrt(2 * len(obs)))
        for (arrival, _), (start, stop) in zip(curves, coverage):
            residuals.extend((10 * max(0, arrival[0] - clip_start - start),
                              10 * max(0, clip_start + stop - arrival[-1])))
        return np.asarray(residuals)

    guess = least_squares(objective, [*point[:2], 0],
                          bounds=([-EDGE_M, -EDGE_M, max(-clock, -90)],
                                  [EDGE_M, EDGE_M, min(86400 - branch_tail - clock, 90)]),
                          x_scale=(1000, 1000, 15), max_nfev=100)
    fitted, sources, curves = objective(guess.x, details=True)
    if not np.isfinite(fitted) or fitted > loss:
        fitted, sources, curves = objective([*point[:2], 0], details=True)
        params = (*point[:2], 0)
    else:
        params = guess.x
    east, north, relative_clock = map(float, params)
    time = clock + relative_clock
    return {"pair_icao": list(pair), "clock_s_from_midnight": time, "coarse_mse_hz2": loss,
            "training_mse_hz2": fitted, "point_enu_m": [east, north, 0],
            "point_lon_lat": coordinates((east, north, 0)), "source_hz": sources}, curves, flights


def locked_predictions(fit, curves, held):
    reports = []
    for slot, ((arrival, ratio), source, blocks) in enumerate(zip(curves, fit["source_hz"], held)):
        entries = []
        for block in blocks:
            times = block[:, 0] + fit["clock_s_from_midnight"]
            covered = times[0] >= arrival[0] and times[-1] <= arrival[-1]
            row = {"clip_time_interval_s": (block[[0, -1], 0] - slot * GAP_S).tolist(),
                   "frames": len(block), "covered": bool(covered), "passes": False}
            if covered:
                predicted = source * np.interp(times, arrival, ratio)
                delta = block[:, 1] - predicted
                error = float(np.median(abs(delta)))
                floor = local_jitter_hz(block, 0.5)
                row.update({"median_absolute_error_hz": error, "local_jitter_floor_hz": floor,
                            "signed_median_error_hz": float(np.median(delta)), "error_to_jitter": error / floor,
                            "passes": error / floor <= SETTINGS["max_error_to_local_jitter"],
                            "samples": np.column_stack((block[:, 0] - slot * GAP_S, block[:, 1], predicted)).tolist()})
            entries.append(row)
        reports.append({"clip": slot + 1, "eligible_blocks": len(entries),
                        "passing_blocks": sum(row["passes"] for row in entries), "blocks": entries})
    return reports


def point_loss(flights, point, traces, coverage, clock):
    curves = [curve(flight, point) for flight in flights]
    lower = max(max(arrival[0] - span[0] for (arrival, _), span in zip(curves, coverage)), 0, clock - 90)
    upper = min(min(arrival[-1] - span[1] for (arrival, _), span in zip(curves, coverage)), 86400 - branch_tail, clock + 90)
    if lower > upper:
        return None
    times = np.linspace(lower, upper, 1 + int(np.ceil((upper - lower) / CLOCK_STEP_S)))
    losses, _ = profile(curves, traces, coverage, times)
    best = losses.argmin()
    value = float(losses[best])
    if len(times) > 1:
        left, right = times[max(best - 1, 0)], times[min(best + 1, len(times) - 1)]
        optimized = minimize_scalar(lambda time: profile(curves, traces, coverage, np.array([time]))[0][0],
                                    bounds=(left, right), method="bounded", options={"xatol": 0.001})
        value = min(value, float(optimized.fun))
    return value if np.isfinite(value) else None


def ring(west, south, east, north):
    return [coordinates((x, y, 0)) for x, y in [(west, south), (east, south), (east, north), (west, north), (west, south)]]


def grid_rows(points, spacing):
    half = spacing / 2
    return [[east, north, up, loss, spacing,
             ring(max(-EDGE_M, east - half), max(-EDGE_M, north - half),
                  min(EDGE_M, east + half), min(EDGE_M, north + half))]
            for east, north, up, loss in points if loss is not None]


def map_grid(coarse, fit, flights, training, coverage, matched):
    stages = [{"spacing_m": 1000, "rows": grid_rows(coarse, 1000),
               "scope": "matched_pair_all_day" if matched else "all_pairs_all_day"}]
    fitted_east, fitted_north, _ = fit["point_enu_m"]
    for spacing, radius in ((100, 1000), (25, 350)):
        east_center = round(fitted_east / spacing) * spacing
        north_center = round(fitted_north / spacing) * spacing
        points = []
        for north in range(max(-EDGE_M, north_center-radius), min(EDGE_M, north_center+radius)+1, spacing):
            for east in range(max(-EDGE_M, east_center-radius), min(EDGE_M, east_center+radius)+1, spacing):
                point = (east, north, 0)
                loss = point_loss(flights, point, training, coverage, fit["clock_s_from_midnight"])
                points.append((*point, loss))
        stages.append({"spacing_m": spacing, "rows": grid_rows(points, spacing), "scope": "locked_winner_segments"})
    return {"cell_row_columns": ["east_m", "north_m", "up_m", "training_mse_hz2",
                                "first_evaluated_spacing_m", "closed_wgs84_lon_lat_ring"],
            "stages": stages}


def feature(geometry_type, coords, **properties):
    return {"type": "Feature", "geometry": {"type": geometry_type, "coordinates": coords}, "properties": properties}


def ranking_row(rank, pair_row):
    pair, (mse, clock, point, hour, first, second) = pair_row
    return {"rank": rank, "pair_icao": list(pair), "mse_hz2": mse, "clock_s_from_midnight": clock,
            "point_enu_m": list(point), "shard_hour_utc": hour, "shard_segment_indices": [first, second]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=Path("runs/livermore-input"))
    parser.add_argument("--archive", type=Path, default=Path("runs/livermore-auto/archive"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matched-pair", action="store_true", help="Restrict aircraft identities; this is not a pair rank")
    parser.add_argument("--reference", type=Path, help="Study GeoJSON; read only after writing the fit lock")
    parser.add_argument("--date", choices=MANIFESTS, default="2026-05-17", help="May 18 is a wrong-day control; use a separate archive cache")
    parser.add_argument("--swap-clips", action="store_true", help="Swap clips, retaining the positive clip-start gap")
    args = parser.parse_args()
    if args.matched_pair and (args.swap_clips or args.date != "2026-05-17"):
        parser.error("--matched-pair is only available for the original order and day")
    day = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc).timestamp()
    clips = CLIPS[::-1] if args.swap_clips else CLIPS
    args.output.mkdir(parents=True, exist_ok=False)

    start = perf_counter()
    audio_receipts = [verify(args.audio / name, digest, frames) for name, digest, frames in clips]
    traces, diagnostics, training, held, coverage = [], [], [], [], []
    for slot, (name, _, _) in enumerate(clips):
        trace, info = harmonic_ridge(args.audio / name)
        train, blocks = split_trace(trace)
        info.update({"fit_frames": len(train), "heldout_blocks": len(blocks),
                     "heldout_frames": sum(map(len, blocks))})
        if len(train) < SETTINGS["minimum_train_observations"] or len(blocks) < SETTINGS["min_heldout_blocks"]:
            raise ValueError(f"Insufficient training or held-out audio in {name}")
        traces.append({"file": name, "time_frequency_rows": trace.tolist(), "diagnostics": info})
        diagnostics.append(info)
        train[:, 0] += slot * GAP_S
        for block in blocks:
            block[:, 0] += slot * GAP_S
        training.append(train)
        held.append(blocks)
        coverage.append((float(trace[0, 0] + slot * GAP_S), float(trace[-1, 0] + slot * GAP_S)))
    write_json(args.output / "automatic-ridges.json", traces)

    rows, receipts, manifest = load_shards(args.archive, day)
    if manifest != MANIFESTS[args.date]:
        raise ValueError(f"Archive differs from the pinned day manifest: {manifest}")
    pair_filter = KNOWN_PAIR if args.matched_pair else None
    shards = [prepare_shard(shard, hour, pair_filter, day) for hour, shard in enumerate(rows)]
    print(f"Prepared {sum(len(flights) for _, flights in shards):,} track segments in {perf_counter() - start:.1f}s", flush=True)
    search_start = perf_counter()
    ranked, coarse = coarse_search(shards, training, coverage, pair_filter)
    rankings = [ranking_row(rank, item) for rank, item in enumerate(ranked, 1)]
    if not ranked:
        raise ValueError("No ordered pair has coverage for both ridges at a shared clock")
    write_json(args.output / "coarse-rankings.json", rankings)
    print(f"Coarse search completed in {perf_counter() - search_start:.1f}s", flush=True)

    refined = [refine(shards, item, training, coverage) for item in ranked[:24]]
    for rank, (fit, _, _) in enumerate(refined, 1):
        fit["coarse_rank"] = rank
    fit, curves, flights = min(refined, key=lambda item: item[0]["training_mse_hz2"])
    controls = sorted((other for other in refined if other[0] is not fit), key=lambda item: item[0]["training_mse_hz2"])[:3]
    mode = "matched_pair" if args.matched_pair else "full_day_all_pairs"
    if args.swap_clips or args.date != "2026-05-17":
        mode = "control_swapped_clips" if args.swap_clips else "control_wrong_day"
    lock = {"mode": mode, "verified": False,
            "reference_used_in_fit": False, "input_date_utc": args.date, "clip_start_gap_s": GAP_S,
            "center_lat_lon_alt": list(origin), "east_north_bounds_m": [-EDGE_M, EDGE_M], "up_m": 0,
            "sound_speed_mps": sound, "coarse_location_step_m": 1000, "coarse_clock_step_s": CLOCK_STEP_S,
            "emission_curve_step_max_s": EMISSION_STEP_S, "audio": audio_receipts, "ridge_diagnostics": diagnostics,
            "archive_manifest_sha256": manifest, "archive_rows_including_shard_overlap": sum(map(len, rows)),
            "archive_hour_receipts": receipts, "guarded_blocks_settings": SETTINGS,
            "unconditioned_distinct_ordered_icao_pairs": None if pair_filter else len(rankings),
            "coarse_top_three": None if pair_filter else rankings[:3],
            "refined_candidate_count": len(refined), "fitted_hypothesis": fit,
            "refined_candidates_in_coarse_rank_order": [candidate[0] for candidate in refined]}
    lock_path = args.output / "fit-lock.json"
    write_json(lock_path, lock)
    lock_hash = sha256(lock_path.read_bytes()).hexdigest()
    print(f"Locked {fit['pair_icao']} at {fit['point_lon_lat']}: {fit['training_mse_hz2']:.6f} Hz^2", flush=True)
    predictions = locked_predictions(fit, curves, held)
    write_json(args.output / "locked-predictions.json", {"fit_lock_sha256": lock_hash,
               "sample_columns": ["clip_relative_time_s", "audio_ridge_hz", "predicted_hz"],
               "winner": predictions,
               "next_best_training_candidates": [{"fit": alternative, "heldout": locked_predictions(alternative, curves, held)}
                                                 for alternative, curves, _ in controls]})

    grid = map_grid(coarse, fit, flights, training, coverage, args.matched_pair)
    reference_distance = None
    if args.reference:
        geojson = json.loads(args.reference.read_text(encoding="utf-8"))
        reference = next(item["geometry"]["coordinates"] for item in geojson["features"]
                         if item.get("properties", {}).get("kind") == "observer")
        reference_distance = great_circle_m(fit["point_lon_lat"][::-1], reference[::-1])
    documented_pair_rank = None if pair_filter else next((entry["rank"] for entry in rankings
                                                          if tuple(entry["pair_icao"]) == KNOWN_PAIR), None)
    marker = feature("Point", fit["point_lon_lat"], verified=False, training_mse_hz2=fit["training_mse_hz2"],
                     distance_to_study_reference_m=reference_distance)
    write_json(args.output / "map.json", {"schema": "livermore-automatic-real-audio-map/v1", "mode": lock["mode"],
               "reference_used_in_fit": False, "verified": False, "fit_lock_sha256": lock_hash,
               "documented_pair_coarse_rank": documented_pair_rank,
               "distinct_ordered_icao_pairs": lock["unconditioned_distinct_ordered_icao_pairs"],
               "distance_to_study_reference_m": reference_distance, "heldout_pass_counts": [
                   [row["passing_blocks"], row["eligible_blocks"]] for row in predictions],
               "search_boundary": feature("Polygon", [ring(-EDGE_M, -EDGE_M, EDGE_M, EDGE_M)]),
               "markers": {"model_best_fit": marker}, "score_grid": grid})
    print(json.dumps({"coarse_documented_pair_rank": documented_pair_rank,
                      "distinct_ordered_icao_pairs": lock["unconditioned_distinct_ordered_icao_pairs"],
                      "reference_distance_m": reference_distance,
                      "heldout_pass_counts": [[row["passing_blocks"], row["eligible_blocks"]] for row in predictions],
                      "elapsed_s": perf_counter() - start}), flush=True)


if __name__ == "__main__":
    main()
