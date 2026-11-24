"""Audio-only initialization for the controlled two-flyover benchmark.

This is a bounded, isolated, descending-tone detector, not a general
classifier. No candidate positions or track times enter detection.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks
from .measurement import fit_ideal_trace
from .search import Event


class InsufficientEvidence(ValueError):
    """The benchmark cannot justify a point estimate."""


def _ridge(evidence, bin_step, frame_step):
    """Viterbi-style maximum-support path with modest slope/continuity cost."""
    n_freq, n_time = evidence.shape
    shifts = range(-round(5 * frame_step / bin_step), round(1.0 * frame_step / bin_step) + 1)
    # Positive delta means a higher current bin than the previous one.
    scores = evidence[:, 0]
    previous = np.empty((n_time, n_freq), dtype=np.int32)
    for col in range(1, n_time):
        options = np.full((len(shifts), n_freq), -np.inf)
        for i, delta in enumerate(shifts):
            start, end = max(0, delta), min(n_freq, n_freq + delta)
            options[i, start:end] = scores[start-delta:end-delta] - 0.16 * abs(delta)
        idx = np.argmax(options, axis=0)
        previous[col] = np.arange(n_freq) - np.asarray(shifts)[idx]
        scores = options[idx, np.arange(n_freq)] + evidence[:, col]
    result = np.empty(n_time, dtype=int)
    result[-1] = np.argmax(scores)
    for col in range(n_time - 1, 0, -1):
        result[col-1] = previous[col, result[col]]
    return result


def _detect_two_events(spec, sound_speed_mps=343.0, *, reject_bad_candidates=False):
    """Recover two supported tone ridges and rough CPAs from this fixture class.

    Fixed assumptions are benchmark-wide: two isolated, descending tones within
    65–195 Hz, visible for roughly 40 s, separated by at least 55 s. These
    assumptions must not be mistaken for universal detector parameters.
    """
    f, audio_db, evidence, dt, peaks = _find_tonal_peaks(spec, 65, 195)
    if len(peaks) != 2 and not reject_bad_candidates:
        raise InsufficientEvidence(f"Expected two isolated supported tones; detected {len(peaks)}")
    events, diagnostics, rejected = [], [], []
    for peak in peaks:
        try:
            event, diag = _characterize_peak(spec, audio_db, evidence, f, dt, peak, sound_speed_mps,
                                             half_window_s=32 if reject_bad_candidates else 24)
        except InsufficientEvidence as error:
            if not reject_bad_candidates:
                raise
            rejected.append({"tonal_peak_audio_s": float(spec.t[peak]), "reason": str(error)})
            continue
        events.append(event)
        diagnostics.append(diag)
    if len(events) != 2:
        error = InsufficientEvidence(f"Expected two supported descending tones; {len(events)} passed among {len(peaks)} peaks")
        error.audit = {"accepted": diagnostics, "rejected": rejected, "raw_peak_count": int(len(peaks))}
        raise error
    return events, diagnostics, {"rejected": rejected, "raw_peak_count": int(len(peaks))}


def _find_tonal_peaks(spec, low_hz, high_hz):
    freq_mask = (spec.f >= low_hz) & (spec.f <= high_hz)
    f = spec.f[freq_mask]
    if len(f) < 4 or len(spec.t) < 4:
        raise InsufficientEvidence("Not enough spectrogram coverage for the benchmark profile")
    audio_db = 10 * np.log10(np.maximum(spec.power[freq_mask], 1e-20))
    # Persistent tones get suppressed by per-bin centering, and impulsive/broad
    # energy is downweighted by comparison with nearby frequency bins.
    local_bg = percentile_filter(audio_db, 55, size=(15, 1), mode="nearest")
    contrast = audio_db - local_bg
    evidence = np.clip(np.minimum(contrast, 3 * spec.whitened[freq_mask]), 0, 26)
    dt = float(np.median(np.diff(spec.t)))
    tonal = gaussian_filter1d(np.mean(np.sort(evidence, axis=0)[-4:], axis=0), 3.0 / dt)
    # Work off the lower half of the tonalness distribution: two long
    # flyovers can occupy more than half of this short fixture.
    base = np.quantile(tonal, 0.25)
    off_peak = tonal[tonal <= np.median(tonal)]
    spread = 1.4826 * np.median(np.abs(off_peak-base))
    peaks, _ = find_peaks(tonal, distance=max(1, round(55/dt)),
                         prominence=max(2.5, 2.8*spread),
                         height=base + max(2.0, 2.2*spread))
    return f, audio_db, evidence, dt, peaks


def _characterize_peak(spec, audio_db, evidence, f, dt, peak, sound_speed_mps, *, half_window_s):
    center = spec.t[peak]
    take = np.where(abs(spec.t-center) <= half_window_s)[0]
    take = take[::max(1, round(0.5/dt))]
    times = spec.t[take]
    if len(times) < 45:
        raise InsufficientEvidence("A detected tone lacks enough recording before/after it")
    path = _ridge(evidence[:, take], float(np.median(np.diff(f))), dt*(take[1]-take[0]))
    support = evidence[path, take]
    # A fitted trajectory cannot make unsupported bins become observations.
    good = support >= 6.0
    if np.mean(good) < 0.70 or good[:len(good)//3].sum() < 8 or good[-len(good)//3:].sum() < 8:
        raise InsufficientEvidence("A tone ridge has too little independent early/late support")
    if f[0] in f[path[good]] or f[-1] in f[path[good]]:
        raise InsufficientEvidence("Tone ridge lies against the benchmark frequency band edge")
    subbins = path.astype(float)
    mid = (path > 0) & (path < len(f)-1)
    k, j = path[mid], take[mid]
    left, top, right = audio_db[k-1, j], audio_db[k, j], audio_db[k+1, j]
    denom = left - 2*top + right
    offset = np.divide(0.5*(left-right), denom, out=np.zeros_like(denom), where=denom < -1e-8)
    subbins[mid] += np.clip(offset, -0.5, 0.5)
    measured = np.interp(subbins, np.arange(len(f)), f)
    t, measured = times[good], measured[good]
    drop_hz = float(np.median(measured[:8]) - np.median(measured[-8:]))
    if drop_hz < 12:
        raise InsufficientEvidence("A detected ridge lacks a sustained downward Doppler change")
    f_mid = float(np.median(measured))
    fits = [fit_ideal_trace(t, measured, (v, r, f_mid, float(center)), sound_speed_mps)
            for v, r in ((70, 1700), (110, 2200), (145, 3000))]
    fit = min(fits, key=lambda a: a["rmse_hz"])
    cpa = fit["delayed_cpa_audio_s"]
    if not fit["success"] or fit["rmse_hz"] > 2.75 or not t[0]+5 < cpa < t[-1]-5:
        raise InsufficientEvidence("Audio-only initial fit is weak or one-sided")
    f0 = fit["f0_hz"]
    event = Event(window_s=(float(t[0]), float(t[-1])), cpa_audio_s=cpa,
                        f0_hz=(max(1, f0-5), f0+5, 0.2),
                        speed_mps=(0, sound_speed_mps), slant_m=(0, 12000),
                        trace_points=np.column_stack((t, measured)).tolist(), trace_sigma_hz=0.75,
                        allowed_engines=("propeller", "jet", "unknown"))
    diagnostics = {"tonal_peak_audio_s":float(center), "supported_fraction":float(np.mean(good)),
                        "drop_hz":drop_hz, "observations":int(len(t)), "initial_ideal_fit":fit}
    return event, diagnostics


def detect_two_events(spec, sound_speed_mps=343.0):
    """Original two-peak profile; intentionally unchanged for the small demo."""
    events, diagnostics, _ = _detect_two_events(spec, sound_speed_mps)
    return events, diagnostics


def detect_two_events_among_distractors(spec, sound_speed_mps=343.0):
    """Use the same evidence gates; reject invalid peaks rather than all input."""
    return _detect_two_events(spec, sound_speed_mps, reject_bad_candidates=True)


FIELD_BANDS_HZ = ((60, 200), (150, 350), (275, 450))


def detect_field_events(spec, sound_speed_mps=343.0, *, edge_aware=True, max_events=2):
    """Select up to two disjoint ridges using audio alone and fixed bands."""
    proposals, rejected = [], []
    for low, high in FIELD_BANDS_HZ:
        f, audio_db, evidence, dt, peaks = _find_tonal_peaks(spec, low, high)
        windows = [(int(peak), 32) for peak in peaks]
        if edge_aware:
            for half_window in (12, 20, 32):
                edge_offset = round(half_window / dt)
                if 2 * edge_offset + 1 < len(spec.t):
                    windows.extend(((edge_offset, half_window), (len(spec.t) - 1 - edge_offset, half_window)))
        for peak, half_window in dict.fromkeys(windows):
            band = {"band_hz": [low, high], "center_audio_s": float(spec.t[peak]), "half_window_s": half_window}
            try:
                event, diag = _characterize_peak(spec, audio_db, evidence, f, dt, peak,
                                                 sound_speed_mps, half_window_s=half_window)
            except InsufficientEvidence as error:
                rejected.append({**band, "reason": str(error)})
                continue
            fit = diag["initial_ideal_fit"]
            quality = diag["supported_fraction"] * diag["drop_hz"] / (fit["f0_hz"] * (1 + fit["rmse_hz"]))
            proposals.append((quality, event, {**diag, **band, "audio_quality": quality}))
    selected = []
    for _, event, diag in sorted(proposals, key=lambda proposal: -proposal[0]):
        if any(abs(event.cpa_audio_s-other.cpa_audio_s) < 55
               or max(event.window_s[0], other.window_s[0]) <= min(event.window_s[1], other.window_s[1])
               for other, _ in selected):
            continue
        selected.append((event, diag))
        if len(selected) == max_events:
            break
    selected.sort(key=lambda item: item[0].cpa_audio_s)
    return [event for event, _ in selected], {
        "selected": [diag for _, diag in selected],
        "supported_proposals": [diag for _, _, diag in proposals],
        "rejected": rejected,
    }
