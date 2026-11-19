"""Bounded candidate search with locked inter-event timing.

The transcript's country-scale index and numerical settings were not supplied.
This reference implementation deliberately exposes those differences.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd
from .physics import closest_approach, forward_curve, interval_closest_range
from .audio import best_frequency_score


@dataclass
class Event:
    window_s: tuple
    cpa_audio_s: float
    f0_hz: tuple
    speed_mps: tuple = (0.0, 300.0)
    slant_m: tuple = (0.0, 10000.0)
    harmonics: tuple = (1,)
    harmonic_weights: tuple = (1.0,)
    trace_points: list | None = None
    trace_sigma_hz: float = 0.5
    allowed_engines: tuple = ("propeller", "unknown")

    def __post_init__(self):
        for span in (self.window_s, self.speed_mps, self.slant_m):
            if len(span) != 2 or not np.all(np.isfinite(span)) or span[1] <= span[0]:
                raise ValueError("Event bounds must be finite and increasing")
        if not self.window_s[0] <= self.cpa_audio_s <= self.window_s[1]:
            raise ValueError("CPA estimate must lie inside the event window")
        if len(self.harmonics) != len(self.harmonic_weights) or not len(self.harmonics):
            raise ValueError("Specify one positive weight per harmonic")
        if any(not np.isfinite(h) or h < 1 or int(h) != h for h in self.harmonics) or any(not np.isfinite(w) or w <= 0 for w in self.harmonic_weights):
            raise ValueError("Harmonics must be positive integers; weights must be positive")
        if not self.allowed_engines or not set(self.allowed_engines) <= {"propeller", "jet", "unknown"}:
            raise ValueError("allowed_engines must use propeller, jet, or unknown")
        self.frequencies()
        if self.trace_points is not None:
            trace = np.asarray(self.trace_points, float)
            if trace.ndim != 2 or trace.shape[1] != 2 or len(trace) < 4 or not np.all(np.isfinite(trace)) or np.any(np.diff(trace[:,0]) <= 0) or not np.isfinite(self.trace_sigma_hz) or self.trace_sigma_hz <= 0 or np.any(trace[:,1] <= 0):
                raise ValueError("Trace requires >=4 ordered [audio_time_s, frequency_hz] points and positive sigma")
            if trace[0,0] > self.window_s[0] or trace[-1,0] < self.window_s[1]:
                raise ValueError("Trace must cover the entire event window")

    def frequencies(self):
        if len(self.f0_hz) != 3:
            raise ValueError("f0_hz must be [minimum, maximum, increment]")
        lo, hi, step = self.f0_hz
        if not np.all(np.isfinite((lo, hi, step))) or not 0 < lo <= hi or step <= 0 or (hi-lo)/step > 10000:
            raise ValueError("Invalid source-frequency sweep")
        return np.arange(lo, hi+step*0.001, step)


@dataclass
class SearchSettings:
    sound_speed_mps: float = 343.0
    offset_tolerance_s: float = 6.0
    start_sweep_s: float = 3.0
    start_step_s: float = 0.25
    curve_step_s: float = 0.25
    known_audio_start_s: float | None = None
    known_start_tolerance_s: float = 1.0
    silence_radius_m: float | None = None
    reject_silence: bool = False

    def __post_init__(self):
        numbers = (self.sound_speed_mps, self.offset_tolerance_s, self.start_sweep_s, self.start_step_s,
                   self.curve_step_s, self.known_start_tolerance_s)
        optional = (self.known_audio_start_s, self.silence_radius_m)
        if not np.all(np.isfinite(numbers)) or any(v is not None and not np.isfinite(v) for v in optional):
            raise ValueError("Search timing, sound speed, and tolerances must be finite")
        if self.silence_radius_m is not None and self.silence_radius_m <= 0:
            raise ValueError("Silence radius must be positive")
        if self.sound_speed_mps <= 0 or self.start_step_s <= 0 or self.curve_step_s <= 0:
            raise ValueError("Positive sound speed and time steps required")
        if min(self.offset_tolerance_s, self.start_sweep_s, self.known_start_tolerance_s) < 0:
            raise ValueError("Tolerances must be nonnegative")
        if self.reject_silence and self.silence_radius_m is None:
            raise ValueError("Hard silence filtering requires an explicit radius")


def _time_offsets(radius, step):
    """Centered lattice including exact tolerance endpoints, even if step is larger."""
    n = int(np.floor(radius / step + 1e-12))
    if n > 10000:
        raise ValueError("Time sweep has more than 20,003 samples")
    points = np.clip(step * np.arange(-n, n+1), -radius, radius)
    if np.isclose(n*step, radius, rtol=1e-12, atol=0):
        return points
    return np.unique(np.r_[-radius, points, radius])


def search(tracks, candidates, events, spec, settings=None, progress=None):
    """One or two events. Candidate xyz and tracks must share a frame.

    Scored outputs are ranking evidence, never posterior probabilities.
    Unknown engine classifications are retained; known jets are filtered only
    from the propeller-tone event match, not the optional silence audit.
    """
    settings = settings or SearchSettings()
    if not 1 <= len(events) <= 2:
        raise ValueError("This version supports one or two flyovers")
    if len(events) == 1 and settings.known_audio_start_s is None:
        raise ValueError("Single-event search requires a known recording start time")
    if len(events) == 2 and events[0].window_s[1] >= events[1].window_s[0]:
        raise ValueError("Two event windows must be ordered and disjoint")
    if len({t.flight_id for t in tracks}) != len(tracks):
        raise ValueError("Each continuous track segment must have a unique flight_id")
    by_id = {t.flight_id: t for t in tracks}
    sampler = spec.sampler()
    rows = []
    audit = {"candidate_points": len(candidates), "tracks": len(tracks),
             "cpa_eligible_pairs": 0, "timing_eligible_pairs": 0,
             "silence_rejected": 0, "scored_pairs": 0, "unsupported_geometry": 0}
    c = settings.sound_speed_mps
    start_offsets = _time_offsets(settings.start_sweep_s, settings.start_step_s)
    known = settings.known_audio_start_s
    if known is not None:
        known_starts = known + _time_offsets(settings.known_start_tolerance_s, settings.start_step_s)
    for point_index, (_, point) in enumerate(candidates.iterrows()):
        observer = point[["x_m", "y_m", "z_m"]].to_numpy(float)
        eligible = [[] for _ in events]
        cpas = {}
        for track in tracks:
            if not any(track.engine.lower() in event.allowed_engines for event in events):
                continue
            cpa = closest_approach(track, observer, c)
            if cpa.at_boundary:
                continue  # Need an extended track rather than guessing a CPA.
            cpas[track.flight_id] = cpa
            for i, event in enumerate(events):
                if track.engine.lower() in event.allowed_engines and event.speed_mps[0] <= cpa.speed_mps <= event.speed_mps[1] and event.slant_m[0] <= cpa.range_m <= event.slant_m[1]:
                    eligible[i].append(track.flight_id)
        curves = {}
        for pair in product(*eligible):
            airframes = {by_id[fid].aircraft_id or fid for fid in pair}
            if len(airframes) != len(pair):
                continue
            audit["cpa_eligible_pairs"] += 1
            if len(events) == 2:
                expected = events[1].cpa_audio_s-events[0].cpa_audio_s
                observed = cpas[pair[1]].arrival_s-cpas[pair[0]].arrival_s
                if abs(observed-expected) > settings.offset_tolerance_s:
                    continue
            center = cpas[pair[0]].arrival_s-events[0].cpa_audio_s
            if settings.known_audio_start_s is None:
                starts = center + start_offsets
            else:
                starts = known_starts[abs(known_starts-center) <= settings.start_sweep_s]
                if not len(starts):
                    continue
            audit["timing_eligible_pairs"] += 1
            try:
                for fid in pair:
                    if fid not in curves:
                        curves[fid] = forward_curve(by_id[fid], observer, c, settings.curve_step_s)
            except ValueError:
                audit["unsupported_geometry"] += 1
                continue
            best = None
            for start in starts:
                scores, fundamentals = zip(*(best_frequency_score(spec, curves[fid], event, start, sampler)
                                             for fid, event in zip(pair, events)))
                if not np.all(np.isfinite(scores)):
                    continue
                total = float(np.sum(scores))
                if best is None or total > best[0]:
                    best = total, float(start), scores, fundamentals
            if best is None:
                continue
            extras = []
            if settings.silence_radius_m is not None and len(events) == 2:
                lo, hi = best[1]+events[0].window_s[1], best[1]+events[1].window_s[0]
                for track in tracks:
                    if track.flight_id in pair:
                        continue
                    try:
                        if interval_closest_range(track, observer, lo, hi, c) < settings.silence_radius_m:
                            extras.append(track.flight_id)
                    except ValueError:
                        pass  # Incomplete geometry is not evidence of an audible flight.
                if settings.reject_silence and extras:
                    audit["silence_rejected"] += 1
                    continue
            row = point.to_dict()
            row.update({"score": best[0], "audio_start_s": best[1],
                        "flight_pair": "|".join(pair), "silence_extra_count": len(extras),
                        "silence_extra_tracks": "|".join(extras)})
            for i, fid in enumerate(pair):
                row.update({f"event_{i+1}_score": float(best[2][i]), f"event_{i+1}_f0_hz": float(best[3][i]),
                            f"event_{i+1}_delayed_cpa_s": cpas[fid].arrival_s})
            rows.append(row)
            audit["scored_pairs"] += 1
        if progress and (point_index % 10 == 0 or point_index == len(candidates)-1):
            progress(f"Scored {point_index+1}/{len(candidates)} candidate points")
    columns = list(candidates.columns) + [
        "score", "audio_start_s", "flight_pair", "silence_extra_count", "silence_extra_tracks",
        *(f"event_{i+1}_{key}" for i in range(len(events)) for key in ("score", "f0_hz", "delayed_cpa_s"))]
    result = pd.DataFrame(rows, columns=columns)
    if len(result):
        result = result.sort_values(["score", "flight_pair", "x_m", "y_m", "z_m"],
                                    ascending=[False, True, True, True, True]).reset_index(drop=True)
    return result, audit


def best_per_pair(results):
    if not len(results):
        return results.copy()
    return results.sort_values("score", ascending=False, kind="stable").drop_duplicates("flight_pair").reset_index(drop=True)


def refine_cartesian(results, radius_m=200, spacing_m=50, top_pairs=2):
    """Refine around each pair winner on a fixed-z Cartesian plane.

    For geography/terrain use fresh DEM-sampled candidate heights instead.
    """
    points = []
    for _, row in best_per_pair(results).head(top_pairs).iterrows():
        for x in np.arange(row.x_m-radius_m, row.x_m+radius_m+spacing_m*0.001, spacing_m):
            for y in np.arange(row.y_m-radius_m, row.y_m+radius_m+spacing_m*0.001, spacing_m):
                points.append((x, y, row.z_m))
    return pd.DataFrame(points, columns=["x_m", "y_m", "z_m"]).drop_duplicates().reset_index(drop=True)
