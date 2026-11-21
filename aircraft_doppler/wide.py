"""Fixed, offline, synthetic-only 10-by-10 km benchmark (no data source or network)."""
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter
import hashlib
import json
import numpy as np
import pandas as pd
from .audio import make_spectrogram, read_audio
from .benchmark import STFT_SETTINGS, XYZ, _write_json
from .detection import InsufficientEvidence, detect_two_events_among_distractors
from .io import load_inputs
from .search import _time_offsets

PROFILE = "synthetic-wide-v2"
SOUND_MPS = 343.
EDGE_M = 5000
SURVEY_M = 500
CLOCK_HALF_S = 80
CLOCK_STEP_S = 1.
LOCAL_HALF_S = 1.
LOCAL_STEP_S = .125
CLOCK_BASINS = 2
TRACE_RMSE_LIMIT_HZ = .85
NEAR_BEST_MSE_HZ2 = .0064
PAIR_MSE_GAP_HZ2 = .0064
MAX_SPAN_M = 400.
# Spatial mode caps apply in addition to retained viable per-pair centers.
STAGES = ((100, 600, 7, 2.0), (25, 250, 6, 0.5))
CLOCKS = np.arange(-CLOCK_HALF_S,CLOCK_HALF_S + CLOCK_STEP_S/2,CLOCK_STEP_S)
LOCAL_OFFSETS = np.arange(-LOCAL_HALF_S,LOCAL_HALF_S + LOCAL_STEP_S/2,LOCAL_STEP_S)
PROGRAM_SETTINGS = {"bounds_xy_m": [[-EDGE_M,EDGE_M],[-EDGE_M,EDGE_M]], "height_m": 0,
                    "survey_spacing_m": SURVEY_M, "clock_global_bounds_s": [-CLOCK_HALF_S,CLOCK_HALF_S],
                    "clock_global_step_s": CLOCK_STEP_S, "clock_global_offsets": len(CLOCKS),
                    "clock_local_half_width_s": LOCAL_HALF_S, "clock_local_step_s": LOCAL_STEP_S,
                    "clock_local_max_basins_per_pair_point": CLOCK_BASINS,
                    "clock_local_clipped_to_global": True,
                    "point_refinements": [{"spacing_m": s,"half_width_m": r,"max_separate_spatial_modes": n,
                                           "per_pair_refine_margin_hz2": m}
                                          for s,r,n,m in STAGES],
                    "sound_speed_mps": SOUND_MPS, "stft": STFT_SETTINGS, "detector_half_window_s": 32,
                    "fit_rmse_limit_hz": TRACE_RMSE_LIMIT_HZ, "near_best_loss_hz2": NEAR_BEST_MSE_HZ2,
                    "pair_gap_hz2": PAIR_MSE_GAP_HZ2, "maximum_support_span_m": MAX_SPAN_M}


class BudgetExceeded(RuntimeError):
    """Cooperative deadline checked between candidate positions, and before each case."""


def _deadline_check(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise BudgetExceeded("Wall-clock budget reached; no unfinished case is counted as verified.")


def _points(rows):
    return [tuple(int(v) for v in row) for row in rows]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_loss(curve, obs, band, offsets, coverage=None):
    """Profile f0 analytically within the audio-only band for each clock."""
    arrival, ratio = curve
    absolute = offsets[:,None]+obs[:,0]
    if coverage is None:
        valid = (absolute[:,0] >= arrival[0]) & (absolute[:,-1] <= arrival[-1])
    else:
        valid = (coverage[0]+offsets >= arrival[0]) & (coverage[1]+offsets <= arrival[-1])
    losses = np.full(len(offsets), np.inf)
    f0 = np.full(len(offsets), np.nan)
    if valid.any():
        q = np.interp(absolute[valid],arrival,ratio)
        f = np.clip(np.einsum('ij,j->i',q,obs[:,1])/np.einsum('ij,ij->i',q,q),*band)
        err = q*f[:,None]-obs[:,1]
        losses[valid] = np.einsum('ij,ij->i',err,err)/obs.shape[0]
        f0[valid] = f
    return losses, f0


def _minima(v):
    left = np.r_[np.inf,v[:-1]]
    right = np.r_[v[1:],np.inf]
    mins = np.flatnonzero(np.isfinite(v) & (v <= left) & (v <= right))
    return mins[np.argsort(v[mins],kind="stable")]


class Profile:
    def __init__(self, tracks, events, deadline=None, *, trace_override=None,
                 clock_center_s=0., clock_half_s=CLOCK_HALF_S):
        if len(events) not in (1, 2):
            raise ValueError("The ridge profile needs one or two events")
        if not np.isfinite(clock_center_s) or not np.isfinite(clock_half_s) or clock_half_s < 0:
            raise ValueError("Invalid clock center or half width")
        self.traces = ([np.asarray(event.trace_points,float) for event in events]
                       if trace_override is None else trace_override)
        self.bands = [e.f0_hz[:2] for e in events]
        self.allowed_engines = [e.allowed_engines for e in events]
        self.coverage = ([None] * len(events) if trace_override is None else [e.window_s for e in events])
        self.clock_center = clock_center_s
        self.clock_half = clock_half_s
        self.clock_lo, self.clock_hi = clock_center_s - clock_half_s, clock_center_s + clock_half_s
        self.clocks = clock_center_s + _time_offsets(clock_half_s, CLOCK_STEP_S)
        self.fields = [*XYZ, "loss_hz2", "flight_pair", "clock_offset_s",
                       *(f"event_{i}_loss_hz2" for i in range(1, len(events) + 1)),
                       *(f"event_{i}_source_hz" for i in range(1, len(events) + 1))]
        self.flights = []
        for track in tracks:
            t = np.linspace(track.t[0],track.t[-1],int(np.ceil((track.t[-1]-track.t[0])/.25))+1)
            velocity = track.velocity(t)
            if (np.linalg.norm(velocity,axis=1) >= SOUND_MPS).any():
                raise ValueError("Model only supports subsonic tracks")
            self.flights.append((track.flight_id, track.aircraft_id or track.flight_id,
                                 track.engine, t, track.position(t), velocity))
        self.cache = {}
        self.deadline = deadline
        self.audit = {"tracks_input":len(self.flights), "positions_scored":0, "pairs_scored":0,
                      "global_clock_pair_hypotheses_with_coverage":0, "local_clock_pair_hypotheses_with_coverage":0}

    def at(self, pos):
        if pos in self.cache:
            return self.cache[pos]
        _deadline_check(self.deadline)
        curves = {}
        options = [{} for _ in self.traces]
        for fid,aid,engine,te,p,v in self.flights:
            # A necessary independent precheck, not dependent on ground truth,
            # a guessed CPA, or a presumed aircraft identity.
            d = p[[0,-1]]-pos
            if not any(start+self.clock_center+self.clock_half >= te[0]+np.linalg.norm(d[0])/SOUND_MPS and
                       stop+self.clock_center-self.clock_half <= te[-1]+np.linalg.norm(d[-1])/SOUND_MPS
                       for obs,cover in zip(self.traces,self.coverage)
                       for start,stop in [cover if cover is not None else (obs[0,0],obs[-1,0])]):
                continue
            delta = p-pos
            slant = np.linalg.norm(delta,axis=1)
            if np.any(slant == 0):
                continue
            ratio = 1/(1+np.einsum('ij,ij->i',delta,v)/(slant*SOUND_MPS))
            curve = te+slant/SOUND_MPS, ratio
            # The curve is monotonically increasing for every subsonic track.
            for i,(obs,band) in enumerate(zip(self.traces,self.bands)):
                if engine not in self.allowed_engines[i]:
                    continue
                total, f = _trace_loss(curve,obs,band,self.clocks,self.coverage[i])
                if np.isfinite(total).any():
                    options[i][fid] = aid,total,f
                    curves[fid] = curve
        rows = []
        for choice in product(*(options_for_event.items() for options_for_event in options)):
            ids, matches = zip(*choice)
            aircraft, losses, sources = zip(*matches)
            if len(set(aircraft)) < len(ids):
                continue
            scores = np.mean(losses, axis=0)
            self.audit["global_clock_pair_hypotheses_with_coverage"] += int(np.isfinite(scores).sum())
            minima = _minima(scores)
            if not len(minima):
                continue
            i = minima[0]
            best = (scores[i], self.clocks[i], *(loss[i] for loss in losses), *(f[i] for f in sources))
            basins = []
            for i in minima:
                clock = self.clocks[i]
                if any(abs(clock-previous) <= 2*LOCAL_HALF_S for previous in basins):
                    continue
                basins.append(clock)
                offsets = clock + LOCAL_OFFSETS
                offsets = offsets[(self.clock_lo <= offsets) & (offsets <= self.clock_hi)]
                profiled = [_trace_loss(curves[fid], trace, band, offsets, coverage)
                            for fid, trace, band, coverage in zip(ids, self.traces, self.bands, self.coverage)]
                losses_local, sources_local = zip(*profiled)
                total = np.mean(losses_local, axis=0)
                self.audit["local_clock_pair_hypotheses_with_coverage"] += int(np.isfinite(total).sum())
                j = int(np.argmin(total))
                if total[j] < best[0]:
                    best = (total[j], offsets[j], *(v[j] for v in losses_local), *(f[j] for f in sources_local))
                if len(basins) == CLOCK_BASINS:
                    break
            rows.append((*pos, best[0], "|".join(ids), *best[1:]))
        self.audit["positions_scored"] += 1
        self.audit["pairs_scored"] += len(rows)
        self.cache[pos] = rows
        return rows

    def table(self, points):
        data = [row for pos in points for row in self.at(pos)]
        return pd.DataFrame(data,columns=self.fields).sort_values(
            ["loss_hz2","x_m","y_m","flight_pair"],kind="stable").reset_index(drop=True)


def _modes(table, points, spacing, limit, radius, pair_margin):
    """Represent separate spatial minima and plausible distinct track pairs."""
    tested = set(points)
    best = table.drop_duplicates(XYZ)
    by_site = {tuple(r[:3]): float(r[3]) for r in best.itertuples(index=False,name=None)}
    leaders = []
    viable = table.drop_duplicates("flight_pair")
    if not viable.empty:
        viable = viable[viable.loss_hz2 <= viable.iloc[0].loss_hz2+pair_margin]
    # A fixed top-pair quota can discard an equally good alternative after
    # rounding on the coarse grid. Always keep each viable pair's best cell
    # unless already safely within a retained refinement square.
    for x,y,z in viable[XYZ].itertuples(index=False,name=None):
        if all(max(abs(x-p[0]),abs(y-p[1])) > max(0,radius-spacing) for p in leaders):
            leaders.append((x,y,z))
    overall_limit = len(leaders)+limit
    for x,y,z in by_site:
        if len(leaders) >= overall_limit:
            break
        q = (x,y,z)
        if not all(np.hypot(x-p[0],y-p[1]) >= 2*radius for p in leaders):
            continue
        around = [(x+dx,y+dy,z) for dx in (-spacing,0,spacing) for dy in (-spacing,0,spacing)
                  if dx or dy if abs(x+dx) <= EDGE_M and abs(y+dy) <= EDGE_M]
        if all(n in tested and by_site.get(n,np.inf) >= by_site[q] for n in around):
            leaders.append(q)
    return leaders


def _refine(centers, spacing, radius):
    return sorted({(x+dx,y+dy,z) for x,y,z in centers
                   for dx in range(-radius,radius+1,spacing) for dy in range(-radius,radius+1,spacing)
                   if abs(x+dx) <= EDGE_M and abs(y+dy) <= EDGE_M})


def _grid(points, score_table):
    table = pd.DataFrame(points,columns=XYZ)
    per_point = score_table.drop_duplicates(XYZ)[XYZ+["loss_hz2"]]
    return table.merge(per_point,on=XYZ,how="left",validate="1:1")


def infer(audio_path, tracks_path, points_path, output, *, deadline=None, hardened=False, channel=0):
    """Inference receives no scene ID, source frequencies, event picks, or truth."""
    start = perf_counter()
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    paths = {"wav":Path(audio_path), "tracks":Path(tracks_path), "candidates":Path(points_path)}
    report = {"profile":"synthetic-wide-v3-blocked" if hardened else PROFILE,"status":"abstained","settings":PROGRAM_SETTINGS,
              "inputs_sha256":{k:_sha256(v) for k,v in paths.items()}}
    def finish():
        report["inference_wall_s"] = round(perf_counter()-start,3)
        _write_json(output/"inference.json",report)
        return report

    sr,audio = read_audio(paths["wav"], channel=channel)
    report["recording_duration_s"] = len(audio)/sr
    spec = make_spectrogram(audio,sr,**STFT_SETTINGS)
    del audio
    tracks,point_table,frame = load_inputs(paths["tracks"],paths["candidates"])
    candidates = _points(point_table[XYZ].to_numpy())
    expected = [(x,y,0) for x in range(-EDGE_M,EDGE_M+1,SURVEY_M) for y in range(-EDGE_M,EDGE_M+1,SURVEY_M)]
    if frame != "cartesian" or set(candidates) != set(expected) or not np.allclose(
            point_table[XYZ],candidates,rtol=0,atol=0):
        raise ValueError("Wide benchmark is fixed to the full synthetic 10 × 10 km lattice")
    _deadline_check(deadline)
    try:
        events, detections, audio_audit = detect_two_events_among_distractors(spec,SOUND_MPS)
    except InsufficientEvidence as exc:
        report.update({"audio_audit":getattr(exc,"audit",{}),"abstention_reasons":[str(exc)]})
        return finish()
    hz_bin = float(np.median(np.diff(spec.f)))
    del spec
    report.update({"audio_only_detection":detections,"audio_audit":audio_audit,
                   "audio_event_separation_s":round(events[1].cpa_audio_s-events[0].cpa_audio_s,3)})
    identifiers = sorted(t.flight_id for t in tracks)
    report["input_track_ids"] = identifiers
    _write_json(output/"detected_events.json",[asdict(e) for e in events])
    use_traces = None
    if hardened:
        from .corroboration import (SETTINGS as HOLD_SETTINGS, MIN_TRAIN_OBSERVATIONS,
                                    MIN_HELDOUT_BLOCKS, split_trace, check_locked_hypothesis)
        report["corroboration_settings"] = {**HOLD_SETTINGS,"require_full_track_coverage_and_disjoint_windows":True}
        if events[0].window_s[1] >= events[1].window_s[0]:
            report["abstention_reasons"] = ["Detected events overlap and are not independent evidence"]
            return finish()
        use_traces,held_blocks = zip(*(split_trace(event.trace_points) for event in events))
        report["corroboration_observations"] = [{"fit_frames":len(fit),"verification_blocks":len(held)}
                                                for fit,held in zip(use_traces,held_blocks)]
        if any(len(fit)<MIN_TRAIN_OBSERVATIONS or len(blocks)<MIN_HELDOUT_BLOCKS
               for fit,blocks in zip(use_traces,held_blocks)):
            report["abstention_reasons"] = ["Insufficient audio for guarded fit and verification"]
            return finish()
    scorer = Profile(tracks,events,deadline,trace_override=use_traces)
    stage_points = [candidates]
    table = scorer.table(candidates)
    _grid(candidates,table).to_csv(output/"grid_500m.csv",index=False)
    report["stage_audit"] = [{"spacing_m":SURVEY_M,"positions":len(candidates),
                              "feasible_positions":len(table.drop_duplicates(XYZ)),
                              "audit_cumulative":dict(scorer.audit)}]
    if table.empty:
        report["abstention_reasons"] = ["No pair of distinct tracks covers both ridges at a common clock offset"]
        return finish()
    for k,(step,radius,limit,pair_margin) in enumerate(STAGES):
        previous_step = SURVEY_M if k == 0 else STAGES[k-1][0]
        reference = set(q for stage in stage_points for q in stage if all(v%previous_step==0 for v in q))
        centers = _modes(table,reference,previous_step,limit,radius,pair_margin)
        points = _refine(centers,step,radius)
        stage = scorer.table(points)
        _grid(points,stage).to_csv(output/f"grid_{step}m.csv",index=False)
        stage_points.append(points)
        all_points = sorted(set(q for group in stage_points for q in group))
        table = scorer.table(all_points)
        report["stage_audit"].append({"spacing_m":step,"positions":len(points),
                                      "feasible_positions":len(stage.drop_duplicates(XYZ)),
                                      "centers_xyz_m":[list(q) for q in centers],
                                      "audit_cumulative":dict(scorer.audit)})

    table.to_csv(output/"final_scores.csv",index=False)
    by_pair = table.drop_duplicates("flight_pair")
    by_pair.to_csv(output/"best_per_pair.csv",index=False)
    win = table.iloc[0]
    near = table[table.loss_hz2 <= win.loss_hz2+NEAR_BEST_MSE_HZ2].drop_duplicates(XYZ)
    span = np.ptp(near[["x_m","y_m"]].to_numpy(float),axis=0)
    edge = bool((abs(near[["x_m","y_m"]]) == EDGE_M).any().any())
    span_too_wide = np.linalg.norm(span) > MAX_SPAN_M
    step = STAGES[-1][0]
    fine = set(stage_points[-1])
    complete = all((p[0]+dx,p[1]+dy,p[2]) in fine for p in near[XYZ].itertuples(index=False,name=None)
                   for dx,dy in ((0,0),(step,0),(-step,0),(0,step),(0,-step)))
    rmses = np.sqrt(win[["event_1_loss_hz2","event_2_loss_hz2"]].to_numpy(float))
    pair_gap = float(by_pair.iloc[1].loss_hz2-win.loss_hz2) if len(by_pair) > 1 else None
    pair_ambiguous = pair_gap is not None and pair_gap < PAIR_MSE_GAP_HZ2
    reasons = []
    if max(rmses) > TRACE_RMSE_LIMIT_HZ:
        reasons.append("At least one event exceeds the absolute ridge RMSE limit")
    if edge:
        reasons.append("Near-best sampled points touch the stated outer boundary")
    if span_too_wide:
        reasons.append("Near-best sampled points span too wide an area for a single point estimate")
    if not complete:
        reasons.append("Near-best sampled points lack a complete final neighborhood")
    if abs(win.clock_offset_s) >= CLOCK_HALF_S:
        reasons.append("Best clock offset reaches the global uncertainty bound")
    report.update({"audit":dict(scorer.audit),
                   "best_fit_diagnostic":{"loss_hz2":float(win.loss_hz2),
                                          "xyz_m":win[XYZ].to_numpy(float).tolist(),
                                          "pair":str(win.flight_pair),"clock_offset_s":float(win.clock_offset_s),
                                          "per_event_rmse_hz":rmses.tolist()},
                   "number_of_feasible_pairs":len(by_pair),
                   "pair_gap_hz2":pair_gap,"pair_ambiguous":bool(pair_ambiguous),
                   "near_best_grid":{"span_xy_m":span.tolist(),"point_count":len(near),
                                     "boundary":edge, "fine_neighborhood_complete":bool(complete)},
                   "abstention_reasons":reasons})
    if hardened:
        report["corroboration"],outside = check_locked_hypothesis(
            win, tracks, use_traces, held_blocks, SOUND_MPS, hz_bin)
        reasons.extend(outside)
        # Make track-identity ambiguity a release gate, not merely a missing tag.
        if pair_ambiguous:
            reasons.append("Near-tied track pairs do not identify the two sources")
    if not reasons:
        if hardened:
            report["evidence_grade"] = "conditional_on_supplied_track_catalogue"
        report.update({"status":"supported_in_this_fixture","estimated_xyz_m":win[XYZ].to_numpy(float).tolist(),
                       "estimated_clock_offset_s":float(win.clock_offset_s)})
        if not pair_ambiguous:
            report["estimated_pair"] = str(win.flight_pair)
    return finish()


def evaluate(report,truth):
    """Must be run separately from inference; adds the synthetic ground truth."""
    point = report.get("estimated_xyz_m")
    expected = truth["expected_outcome"]
    truth_xyz = truth["stationary_virtual_receiver_xyz_m"]
    err = float(np.linalg.norm(np.array(point)-truth_xyz)) if point is not None and truth_xyz is not None else None
    clock_err = abs(report["estimated_clock_offset_s"]-truth["clock_offset_s"]) if point is not None else None
    return {**report,"truth_used_for_inference":False,"truth":truth,"position_error_m":err,
            "clock_absolute_error_s":clock_err,"correct_track_pair":report.get("estimated_pair")==truth["correct_flight_pair"]
            if point is not None else None,
            "expected_behavior_observed":(point is None) if expected=="abstain" else bool(
                err is not None and err <= 150 and report.get("estimated_pair")==truth["correct_flight_pair"])}


def run(output,*,case_slugs=None,max_seconds=540,hardened=False,progress=print):
    from .wide_fixture import SCENES,make_scene
    from .wide_plot import plot_benchmark, plot_timeline
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose a new/empty directory to prevent mixing runs")
    scene_list = [scene for scene in SCENES if (scene.slug in case_slugs if case_slugs else scene.split != "pilot")]
    slugs = [scene.slug for scene in scene_list]
    if case_slugs and not set(case_slugs).issubset(slugs):
        raise ValueError("Unrecognized synthetic fixture name")
    t0 = perf_counter()
    deadline = t0+max_seconds
    run_summary = {"profile":"synthetic-wide-v3-blocked" if hardened else PROFILE,"status":"running","planned_cases":slugs,"cases_completed":{},
                   "max_wall_seconds":max_seconds,"settings":PROGRAM_SETTINGS}

    def checkpoint():
        run_summary["elapsed_wall_s"] = round(perf_counter()-t0,3)
        tmp = output/"summary.tmp"
        _write_json(tmp,run_summary)
        tmp.replace(output/"suite_summary.json")

    checkpoint()
    for scene in scene_list:
        try:
            _deadline_check(deadline)
            start = perf_counter()
            folder = output/scene.slug
            inputs = folder/"inputs"
            progress(f"{scene.slug}: generating 27-minute synthetic WAV…")
            make_scene(inputs,scene)
            _deadline_check(deadline)
            result = infer(inputs/"synthetic.wav",inputs/"tracks.csv",inputs/"candidates.csv",folder/"results",
                           deadline=deadline,hardened=hardened)
            truth = json.loads((inputs/"truth.json").read_text(encoding="utf-8"))
            summary = evaluate(result,truth)
            summary["case_wall_s"] = round(perf_counter()-start,3)
            _write_json(folder/"results"/"summary.json",summary)
            run_summary["cases_completed"][scene.slug] = summary
            checkpoint()
            point = summary["position_error_m"]
            outcome = f"{point:.1f} m position error" if point is not None else "abstained"
            progress(f"{scene.slug}: {outcome} in {summary['case_wall_s']:.1f} s")
        except BudgetExceeded:
            run_summary["status"] = "timed_out_partial"
            run_summary["first_unfinished_case"] = scene.slug
            break
    else:
        run_summary["status"] = "completed"
    checkpoint()
    if run_summary["cases_completed"] and perf_counter() < deadline:
        plot_benchmark(output,run_summary)
        if perf_counter() < deadline:
            plot_timeline(output,run_summary)
    checkpoint()
    return run_summary
