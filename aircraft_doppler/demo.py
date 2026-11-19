"""Independent analytic renderer plus a small, reproducible synthetic search.

Renderer solves the retarded-time equation by Newton iteration and evaluates
source phase. Estimator instead fits sampled tracks and maps emission times
forward. A self-consistency test is useful, but is not a real-world benchmark.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import wavfile
from .audio import make_spectrogram, save_spectrogram
from .io import load_tracks, load_config
from .search import search, best_per_pair, refine_cartesian


def analytic_flight(t, event):
    u = np.asarray(t)-[80, 1000][event]
    if event == 0:
        p = np.stack((90*u+0.025*u*u, 1000+160*np.sin(u/40), 1700+u), axis=-1)
        v = np.stack((90+0.05*u, 4*np.cos(u/40), np.ones_like(u)), axis=-1)
    else:
        p = np.stack((1800+70*np.sin(u/38), -100*u+0.02*u*u, 1500-0.4*u), axis=-1)
        v = np.stack((70/38*np.cos(u/38), -100+0.04*u, np.full_like(u, -0.4)), axis=-1)
    return p, v


def render(seed=7, observer=(137.0, -83.0, 0.0), noise_sd=0.08, c=343.0,
           source_frequencies=(130.35, 102.15), duration_s=1120, sr=2000, centers=(80, 1000)):
    rng = np.random.default_rng(seed)
    t = np.arange(round(duration_s*sr))/sr
    audio = rng.normal(0, noise_sd, len(t)) + 0.08*np.sin(2*np.pi*60*t)
    for event, center in enumerate(centers):
        shift = (80, 1000)[event] - center
        use = (t > center-65) & (t < center+85)
        tr = t[use]
        te = tr-7.0
        for _ in range(12):
            p, v = analytic_flight(te+shift, event)
            delta = p-np.asarray(observer)
            r = np.linalg.norm(delta, axis=1)
            rr = np.sum(delta*v, axis=1)/r
            te -= (te+r/c-tr)/(1+rr/c)
        residual = te+np.linalg.norm(analytic_flight(te+shift, event)[0]-observer, axis=1)/c-tr
        if np.max(abs(residual)) > 1e-8:
            raise RuntimeError("Independent acoustic renderer did not converge")
        phase = 2*np.pi*source_frequencies[event]*te
        envelope = np.exp(-((te-center)/65)**4)*2000/r
        audio[use] += envelope*(0.055*np.sin(phase)+0.028*np.sin(2*phase+0.6))
    return sr, audio.astype(np.float32)


def create_inputs(directory, seed=7, observer=(137.0, -83.0, 0.0), noise_sd=0.08, render_c=343.0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sr, audio = render(seed, observer, noise_sd, render_c)
    wavfile.write(directory/"synthetic.wav", sr, audio)
    rows = []
    # Sampled at 2 s, whereas the renderer uses analytic continuous tracks.
    for event, center in enumerate((80, 1000)):
        times = np.arange(center-80, center+80.01, 2.0)
        p, _ = analytic_flight(times, event)
        for t, xyz in zip(times, p):
            rows.append((f"synthetic_{event+1}", t, *xyz, "propeller"))
        # One wrong-shape first flight and one incorrectly timed second flight.
        if event == 0:
            decoy = p.copy()
            decoy[:,1] = 1000+0.5*(times-center)
            for t, xyz in zip(times, decoy):
                rows.append(("decoy_shape", t, *xyz, "unknown"))
        else:
            for t, xyz in zip(times+12.0, p):
                rows.append(("decoy_time", t, *xyz, "propeller"))
    pd.DataFrame(rows, columns=["flight_id", "time_s", "x_m", "y_m", "z_m", "engine"]).to_csv(directory/"tracks.csv", index=False)
    xx, yy = np.meshgrid(np.arange(-600, 601, 200), np.arange(-600, 601, 200))
    pd.DataFrame({"x_m": xx.ravel(), "y_m": yy.ravel(), "z_m": 0.0}).to_csv(directory/"candidates.csv", index=False)
    config = {"events": [
        {"window_s": [50,125], "cpa_audio_s": 87.2, "f0_hz": [129,132,0.25],
         "speed_mps": [80,102], "slant_m": [1500,3000], "harmonics": [1,2], "harmonic_weights": [1,0.5]},
        {"window_s": [970,1045], "cpa_audio_s": 1007.5, "f0_hz": [101,104,0.25],
         "speed_mps": [88,110], "slant_m": [1500,3000], "harmonics": [1,2], "harmonic_weights": [1,0.5]}],
        "search": {"offset_tolerance_s": 8, "start_sweep_s": 4, "start_step_s": 0.5},
        "stft": {"window_s": 2.0, "hop_s": 0.25, "fmin": 40, "fmax": 500}}
    (directory/"config.json").write_text(json.dumps(config, indent=2))
    truth = {"kind": "SYNTHETIC ONLY", "seed": seed, "observer_xyz_m": list(observer),
             "audio_start_s": 0.0, "render_sound_speed_mps": render_c,
             "noise_sd": noise_sd, "source_frequencies_hz": [130.35,102.15],
             "correct_flight_pair": "synthetic_1|synthetic_2"}
    (directory/"truth.json").write_text(json.dumps(truth, indent=2))
    return sr, audio, truth


def run_demo(directory, progress=print):
    directory = Path(directory)
    sr, audio, truth = create_inputs(directory)
    events, settings, stft_settings = load_config(directory/"config.json")
    spec = make_spectrogram(audio, sr, **stft_settings)
    save_spectrogram(spec, directory/"spectrogram.npz")
    tracks = load_tracks(directory/"tracks.csv")
    coarse, audit = search(tracks, pd.read_csv(directory/"candidates.csv"), events, spec, settings, progress)
    if not len(coarse):
        raise RuntimeError("No synthetic matches survived")
    coarse.to_csv(directory/"coarse_scores.csv", index=False)
    refined = refine_cartesian(coarse, radius_m=200, spacing_m=50, top_pairs=1)
    refined.to_csv(directory/"refined_candidates.csv", index=False)
    settings.start_step_s = 0.25
    fine, fine_audit = search(tracks, refined, events, spec, settings, progress)
    fine.to_csv(directory/"scores.csv", index=False)
    best_per_pair(fine).to_csv(directory/"best_per_pair.csv", index=False)
    winner = fine.iloc[0]
    error = float(np.linalg.norm(winner[["x_m","y_m","z_m"]].to_numpy(float)-truth["observer_xyz_m"]))
    summary = {"status": "synthetic validation only; no field accuracy claim", "position_error_m": error,
               "winning_pair": winner.flight_pair, "correct_pair": winner.flight_pair == truth["correct_flight_pair"],
               "estimated_audio_start_s": winner.audio_start_s,
               "winner_xyz_m": [float(winner.x_m),float(winner.y_m),float(winner.z_m)],
               "coarse_audit": audit, "fine_audit": fine_audit}
    (directory/"summary.json").write_text(json.dumps(summary, indent=2))
    plot_result(spec, tracks, events, fine, directory/"validation.png", truth)
    return summary


def plot_result(spec, tracks, events, results, output, truth=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .physics import forward_curve
    winner = results.iloc[0]
    observer = winner[["x_m","y_m","z_m"]].to_numpy(float)
    ids = winner.flight_pair.split("|")
    by_id = {t.flight_id:t for t in tracks}
    fig, axes = plt.subplots(1, len(events)+1, figsize=(6*(len(events)+1),5), constrained_layout=True)
    for i,event in enumerate(events):
        ax = axes[i]
        use = (spec.t >= event.window_s[0])&(spec.t <= event.window_s[1])
        f0 = winner[f"event_{i+1}_f0_hz"]
        f_use = (spec.f > f0*0.65)&(spec.f < f0*1.4)
        ax.pcolormesh(spec.t[use],spec.f[f_use],spec.whitened[f_use][:,use],shading="auto",cmap="magma",vmin=-1,vmax=10)
        curve = forward_curve(by_id[ids[i]], observer)
        ax.plot(curve.arrival_s-winner.audio_start_s,curve.ratio*f0,color="#53e7ff",linewidth=1,label="Recovered model")
        ax.set(xlim=event.window_s,ylim=(spec.f[f_use][0],spec.f[f_use][-1]),xlabel="Audio time (s)",ylabel="Frequency (Hz)",title=f"Synthetic flyover {i+1}")
        ax.legend(loc="upper right")
    ax = axes[-1]
    spatial = results.groupby(["x_m","y_m"],as_index=False).score.max()
    p = ax.scatter(spatial.x_m,spatial.y_m,c=spatial.score,cmap="viridis",s=60)
    ax.scatter([winner.x_m],[winner.y_m],marker="x",c="red",s=100,label="Recovered")
    if truth:
        ax.scatter(*np.asarray(truth["observer_xyz_m"])[:2,None],marker="*",c="orange",s=150,label="Synthetic truth")
    ax.set(xlabel="Local x (m)",ylabel="Local y (m)",title="Ranking score, not confidence",aspect="equal")
    ax.legend()
    fig.colorbar(p,ax=ax,label="Combined brightness score")
    fig.suptitle("SYNTHETIC BENCHMARK: sampled tracks vs independently rendered audio",fontsize=14)
    fig.savefig(output,dpi=150)
    plt.close(fig)
