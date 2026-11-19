"""Guarded temporal validation of locked synthetic track/location/clock hypotheses."""
import numpy as np
from .physics import forward_curve

PERIOD_S = 16.0
BLOCK_LO_S = 6.0
BLOCK_HI_S = 10.0
GUARD_S = 2.0    # Match the audio Hann window; no overlapping train/validation STFT windows.
MIN_BLOCK_OBSERVATIONS = 5
MIN_HELDOUT_BLOCKS = 3
MIN_TRAIN_OBSERVATIONS = 32
MAX_NOISE_MULTIPLIER = 3.0
MIN_PASS_FRACTION = .75
BIN_FLOOR_FRACTION = .125
SETTINGS = {"period_s": PERIOD_S, "holdout_phase_s":[BLOCK_LO_S,BLOCK_HI_S],
            "guard_s":GUARD_S, "min_measured_frames_per_block":MIN_BLOCK_OBSERVATIONS,
            "min_heldout_blocks":MIN_HELDOUT_BLOCKS, "minimum_train_observations":MIN_TRAIN_OBSERVATIONS,
            "max_error_to_local_jitter":MAX_NOISE_MULTIPLIER,"minimum_passing_fraction":MIN_PASS_FRACTION,
            "stft_bin_floor_fraction":BIN_FLOOR_FRACTION}


def split_trace(values):
    """Disjoint fit/verification samples with a one-full-Hann-window guard."""
    trace = np.asarray(values,dtype=float)
    elapsed = trace[:,0] - trace[0,0]
    phase = np.remainder(elapsed, PERIOD_S)
    train_mask = ((phase <= BLOCK_LO_S-GUARD_S) | (phase >= BLOCK_HI_S+GUARD_S))
    use = ((phase >= BLOCK_LO_S) & (phase <= BLOCK_HI_S))
    ids = np.floor(elapsed / PERIOD_S).astype(int)
    held = [trace[use & (ids == i)] for i in np.unique(ids[use])]
    return trace[train_mask], [block for block in held if len(block) >= MIN_BLOCK_OBSERVATIONS]


def local_jitter_hz(block, bin_spacing):
    """Estimate high-frequency ridge scatter, not smooth curvature or bias.

    Standard deviation of [1,-2,1] differences for independent errors is
    sqrt(6)*sigma. Overlapping STFT windows correlate samples, so this is a
    noise proxy, not a statistical standard error. Keep a sub-bin floor.
    """
    dt = np.diff(block[:,0])
    adjacent = np.isclose(dt[:-1], dt[1:], rtol=1e-6,atol=1e-5) & (dt[:-1] <= .55)
    differences = np.diff(block[:,1],n=2)[adjacent]
    jitter = 0.
    if len(differences) >= 3:
        jitter = 1.4826*np.median(abs(differences-np.median(differences)))/np.sqrt(6)
    return max(float(bin_spacing*BIN_FLOOR_FRACTION),float(jitter))


def check_locked_hypothesis(winner, tracks, train, held, sound_speed, bin_spacing):
    """Never refit the clock/location/frequencies against withheld samples."""
    pair = winner.flight_pair.split("|")
    table = {track.flight_id:track for track in tracks}
    pos = winner[["x_m","y_m","z_m"]].to_numpy(float)
    clock = float(winner.clock_offset_s)
    reports = []
    reasons = []
    for i,(track_id,training,blocks) in enumerate(zip(pair,train,held),1):
        details = {"event":i,"fitted_observations":len(training),"blocks":[]}
        reports.append(details)
        if len(training) < MIN_TRAIN_OBSERVATIONS or len(blocks) < MIN_HELDOUT_BLOCKS:
            reasons.append(f"Event {i} lacks enough guarded corroboration blocks")
            continue
        curve = forward_curve(table[track_id],pos,sound_speed,step_s=.25)
        f0 = float(winner[f"event_{i}_source_hz"])
        for block in blocks:
            ts = block[:,0]+clock
            covered = bool(ts[0] >= curve.arrival_s[0] and ts[-1] <= curve.arrival_s[-1])
            row = {"audio_interval_s":[float(block[0,0]),float(block[-1,0])],
                   "observations":len(block),"covered":covered,"passes":False}
            if covered:
                ratio = np.interp(ts,curve.arrival_s,curve.ratio)
                delta = block[:,1]-f0*ratio
                scale = local_jitter_hz(block,bin_spacing)
                residual = float(np.median(abs(delta)))
                row.update({"median_absolute_error_hz":residual,"local_jitter_floor_hz":scale,
                            "signed_median_error_hz":float(np.median(delta)),
                            "error_to_jitter":residual/scale,"passes":bool(residual/scale <= MAX_NOISE_MULTIPLIER)})
            details["blocks"].append(row)
        fraction = sum(row["passes"] for row in details["blocks"])/len(blocks)
        details["passing_fraction"] = fraction
        if fraction < MIN_PASS_FRACTION:
            reasons.append(f"Event {i} failed locked predictions on withheld audio ({fraction:.0%} blocks passed)")
    return reports,reasons
