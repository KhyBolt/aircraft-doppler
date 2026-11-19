"""Deterministic, offline benchmark with no manually supplied event picks.

Inputs to the estimator are only the WAV, synthetic tracks, and search grid.
Synthetic truth is read by the evaluation step after inference is finalized.
"""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import wavfile
from .audio import make_spectrogram, read_audio
from .detection import InsufficientEvidence, detect_two_events
from .demo import analytic_flight, render
from .io import load_inputs
from .physics import forward_curve
from .search import SearchSettings, best_per_pair, search


PROFILE_VERSION = "synthetic-tone-benchmark-v1"
FIT_RMSE_LIMIT_HZ = 0.85
PAIR_SCORE_MARGIN = 0.025
SUPPORT_LOSS = 0.035
MAX_SUPPORT_SPAN_M = 300.0
XYZ = ["x_m", "y_m", "z_m"]
STFT_SETTINGS = {"window_s": 2.0, "hop_s": 0.25, "fmin": 40, "fmax": 450}
COARSE_SETTINGS = SearchSettings(offset_tolerance_s=7, start_sweep_s=5, start_step_s=0.5)


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def make_fixture(path, *, seed=73, observer=(117., -183., 0.), noise=0.092, render_c=343.):
    """Regenerate original, from-scratch CC0 fixture; not needed when replaying it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    centers = (80., 230.)
    frequencies = (127.7, 105.9)
    sr, audio = render(seed, observer, noise, render_c, frequencies, duration_s=310, centers=centers)
    wavfile.write(path/"synthetic.wav", sr, np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2"))
    rows = []
    for i, center in enumerate(centers):
        ts = center + np.arange(-80, 81, 2, dtype=float)
        positions, _ = analytic_flight(ts+(80, 1000)[i]-center, i)
        fid = ("flyover_a", "flyover_b")[i]
        rows.extend((fid, fid, t, *p, "propeller") for t, p in zip(ts, positions))
        if i == 0:
            # A plausible earlier wrong-shape flight with a different turn.
            decoy = positions.copy()
            decoy[:, 1] = 665 - 0.024*(ts-center)**2
            decoy[:, 2] += 340
            rows.extend(("decoy_shape", "decoy_shape", t-1.8, *p, "unknown") for t, p in zip(ts, decoy))
        else:
            rows.extend(("decoy_time", "decoy_time", t+12., *p, "propeller") for t, p in zip(ts, positions))
    pd.DataFrame(rows, columns=["flight_id", "aircraft_id", "time_s", *XYZ, "engine"]).to_csv(path/"tracks.csv", index=False)
    grid = [(x, y, 0) for x in range(-600, 601, 200) for y in range(-600, 601, 200)]
    pd.DataFrame(grid, columns=XYZ).to_csv(path/"candidates.csv", index=False)
    _write_json(path/"truth.json", {"kind": "synthetic only", "seed": seed, "observer_xyz_m": list(observer),
                                    "audio_start_s": 0., "render_sound_speed_mps": render_c, "noise_sd": noise,
                                    "source_frequencies_hz": list(frequencies), "correct_flight_pair": "flyover_a|flyover_b"})


def _residuals(winner, events, tracks, c):
    observer = winner[XYZ].to_numpy(float)
    by_id = {track.flight_id: track for track in tracks}
    output = []
    for i, (fid, event) in enumerate(zip(winner.flight_pair.split("|"), events), 1):
        trace = np.asarray(event.trace_points, float)
        curve = forward_curve(by_id[fid], observer, c)
        predicted = winner[f"event_{i}_f0_hz"] * np.interp(trace[:, 0]+winner.audio_start_s, curve.arrival_s, curve.ratio)
        output.append(float(np.sqrt(np.mean((trace[:, 1]-predicted)**2))))
    return output


def _profile_grid(candidates, scores):
    by_point = scores.groupby(XYZ, as_index=False).score.max().rename(columns={"score":"profile_score"})
    return candidates[XYZ].drop_duplicates().merge(by_point, on=XYZ, how="left", validate="1:1")


def _interior_candidates(results, domain):
    """Refine the two best pairs, plus up to two distinct near-best modes."""
    centers = []
    pairs = best_per_pair(results).head(2)
    for row in pairs.itertuples(index=False):
        centers.append(np.array([row.x_m, row.y_m, row.z_m]))
    for row in results.itertuples(index=False):
        center = np.array([row.x_m, row.y_m, row.z_m])
        if row.score < results.iloc[0].score - 0.12 or len(centers) >= 4:
            break
        if all(np.linalg.norm(center-p) > 250 for p in centers):
            centers.append(center)
    # np.arange here is integral; this keeps the boundary inside the stated ROI.
    shifts = np.arange(-200, 201, 50)
    points = pd.DataFrame([center + [dx,dy,0] for center in centers for dx in shifts for dy in shifts], columns=XYZ)
    within = ((points[XYZ] >= domain[XYZ].min()) & (points[XYZ] <= domain[XYZ].max())).all(axis=1)
    return points.loc[within].drop_duplicates().reset_index(drop=True)


def infer(audio_path, tracks_path, candidates_path, output):
    """Run the offline protocol. Deliberately accepts no path to fixture truth."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new or empty output directory to avoid mixing stages from different runs")
    paths = {"audio": Path(audio_path), "tracks": Path(tracks_path), "candidates": Path(candidates_path)}
    report = {"benchmark_profile": PROFILE_VERSION, "status": "abstained", "scope": "synthetic benchmark only",
              "caution": "Model fit and grid support are not probabilities or calibrated location uncertainty.",
              "inputs_sha256": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in paths.items()},
              "stft": STFT_SETTINGS, "coarse_search": asdict(COARSE_SETTINGS),
              "decision_thresholds": {"fit_rmse_limit_hz": FIT_RMSE_LIMIT_HZ, "pair_score_margin": PAIR_SCORE_MARGIN,
                                      "support_score_loss": SUPPORT_LOSS, "max_support_span_m": MAX_SUPPORT_SPAN_M}}
    sr, x = read_audio(audio_path)
    spec = make_spectrogram(x, sr, **STFT_SETTINGS)
    tracks, candidates, frame = load_inputs(tracks_path, candidates_path)
    if not np.all(candidates.z_m == candidates.z_m.iloc[0]) or frame != "cartesian":
        raise ValueError("The fixture benchmark uses a local fixed-height Cartesian grid")
    xy = candidates[["x_m", "y_m"]]
    scaled = ((xy-xy.min())/200).to_numpy()
    if any(xy.nunique() < 2) or not np.allclose(scaled, np.rint(scaled), atol=1e-9, rtol=0):
        raise ValueError("The fixture benchmark requires a 200 m Cartesian lattice with at least two values per axis")
    try:
        events, report["audio_only_detection"] = detect_two_events(spec, COARSE_SETTINGS.sound_speed_mps)
    except InsufficientEvidence as error:
        report["abstention_reasons"] = [str(error)]
        _write_json(output/"inference.json", report)
        return report
    _write_json(output/"detected_events.json", [asdict(event) for event in events])
    coarse, report["coarse_audit"] = search(tracks, candidates, events, spec, COARSE_SETTINGS)
    coarse.to_csv(output/"coarse_scores.csv", index=False)
    _profile_grid(candidates, coarse).to_csv(output/"coarse_grid.csv", index=False)
    if coarse.empty:
        report["abstention_reasons"] = ["No joint fit survived the two-event shared-clock and coverage checks"]
        _write_json(output/"inference.json", report)
        return report
    refined = _interior_candidates(coarse, candidates)
    settings = replace(COARSE_SETTINGS, start_step_s=0.125)
    fine_events = [replace(e, f0_hz=(*e.f0_hz[:2], 0.05)) for e in events]
    # Revisit every coarse location at the *same* frequency/clock resolution
    # as the local fine grid before calculating score differences.
    final_points = pd.concat([refined, candidates[XYZ]], ignore_index=True).drop_duplicates(XYZ)
    evaluated, report["final_audit"] = search(tracks, final_points, fine_events, spec, settings)
    report["fine_search"] = asdict(settings)
    report["refined_location_count"] = len(refined)
    report["coarse_grid_note"] = "Coarse locations re-evaluated using final nuisance parameter resolution."
    evaluated.to_csv(output/"final_scores.csv", index=False)
    _profile_grid(refined, evaluated).to_csv(output/"fine_grid.csv", index=False)
    _profile_grid(candidates, evaluated).to_csv(output/"coarse_grid.csv", index=False)
    pairs = best_per_pair(evaluated)
    pairs.to_csv(output/"best_per_pair.csv", index=False)
    winner = evaluated.iloc[0]
    residuals = _residuals(winner, fine_events, tracks, settings.sound_speed_mps)
    near = evaluated[evaluated.score >= winner.score - SUPPORT_LOSS].drop_duplicates(XYZ)
    span = np.ptp(near[XYZ].to_numpy(float), axis=0)
    reached_edge = any((near[axis] <= candidates[axis].min()).any() or (near[axis] >= candidates[axis].max()).any()
                       for axis in ("x_m", "y_m"))
    pair_margin = None if len(pairs) < 2 else float(winner.score-pairs.iloc[1].score)
    reasons = []
    if max(residuals) > FIT_RMSE_LIMIT_HZ:
        reasons.append("At least one event exceeds the absolute observed-ridge RMSE limit")
    if reached_edge:
        reasons.append("Near-best score support touches the outer grid boundary")
    if np.linalg.norm(span) > MAX_SUPPORT_SPAN_M:
        reasons.append("Near-best grid points span too large an area for a point estimate")
    fine_sites = set(refined.itertuples(index=False, name=None))
    if any((row.x_m+dx,row.y_m+dy,row.z_m) not in fine_sites for row in near.itertuples()
           for dx,dy in ((0,0), (50,0), (-50,0), (0,50), (0,-50))):
        reasons.append("Near-best location lacks a complete fine-grid neighborhood")
    report.update({"status":"abstained" if reasons else "supported_in_this_fixture",
                   "abstention_reasons":reasons,
                   "best_score_diagnostic":float(winner.score),
                   "per_event_fit_rmse_hz":residuals,
                   "runner_up_pair_score_margin":pair_margin,
                   "pair_status":"ambiguous" if pair_margin is not None and pair_margin < PAIR_SCORE_MARGIN else
                                 ("only_one_feasible_pair" if pair_margin is None else "separated_in_this_fixture"),
                   "near_best_sampled_grid": {"point_count":len(near), "span_xy_m":span[:2].tolist(),
                                              "touches_outer_boundary":bool(reached_edge),
                                              "note":"Score-loss threshold on sampled grid; not an accuracy or confidence interval."}})
    if not reasons:
        report.update({"estimated_xyz_m":[float(winner[key]) for key in XYZ],
                       "estimated_audio_start_s":float(winner.audio_start_s)})
        if report["pair_status"] != "ambiguous":
            report["estimated_pair"] = winner.flight_pair
    _write_json(output/"inference.json", report)
    return report


def benchmark(fixture, output):
    """Evaluate after inference, then render the score grids with truth overlaid."""
    fixture, output = Path(fixture), Path(output)
    report = infer(fixture/"synthetic.wav", fixture/"tracks.csv", fixture/"candidates.csv", output)
    truth = json.loads((fixture/"truth.json").read_text(encoding="utf-8"))
    report = {**report, "truth_used_for_inference":False, "synthetic_truth_xyz_m":truth["observer_xyz_m"],
              "synthetic_truth_pair":truth["correct_flight_pair"]}
    if "estimated_xyz_m" in report:
        report["position_error_m"] = float(np.linalg.norm(np.array(report["estimated_xyz_m"])-truth["observer_xyz_m"]))
    if "estimated_pair" in report:
        report["correct_pair"] = report["estimated_pair"] == truth["correct_flight_pair"]
    _write_json(output/"summary.json", report)
    if (output/"fine_grid.csv").is_file():
        from .heatmap import plot_score_heatmap
        plot_score_heatmap(pd.read_csv(output/"coarse_grid.csv"), pd.read_csv(output/"fine_grid.csv"),
                           report, output/"score_heatmap.png")
    return report
