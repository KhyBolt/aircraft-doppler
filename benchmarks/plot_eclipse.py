"""Render an evaluated field replay with the official USGS Dayton basemap."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from aircraft_doppler.audio import make_spectrogram, read_audio
from aircraft_doppler.benchmark import STFT_SETTINGS, XYZ
from aircraft_doppler.wide import EDGE_M
from benchmarks.eclipse_replay import LAT, to_geographic

BASEMAP_EXTENT = (-84.28, -84.12, 39.67, 39.79)
TITLES = {"registered": "Registered window", "extended": "Extended window",
          "adjacent": "Disjoint comparison", "wrong-time": "Wrong-time control",
          "wrong-audio": "Wrong-audio control"}


def draw(output, basemap, destination):
    infer = json.loads((output / "inference.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    ridges = json.loads((output / "ridges.json").read_text(encoding="utf-8"))
    fig, (map_ax, audio_ax) = plt.subplots(1, 2, figsize=(12.4, 6.5), gridspec_kw={"width_ratios": [1.07, 1]})
    fig.patch.set_facecolor("#f7f8fa")
    fig.subplots_adjust(left=.08, right=.96, bottom=.18, top=.79, wspace=.28)
    fig.text(.075, .93, "Eclipse Soundscapes 505", fontsize=24, weight="bold", color="#152e36")
    title = TITLES[infer["arm"]]
    status = {"abstained": "Solver abstained", "timed_out": "Time budget reached",
              "supported_under_test_profile": "Estimate passed this profile's gates"}[infer["status"]]
    fig.text(.077, .87, f"{title}   |   {infer['duration_s'] / 60:g} minutes   |   {status}",
             fontsize=12, color="#334e56")

    map_ax.imshow(plt.imread(basemap), extent=BASEMAP_EXTENT, alpha=.87)
    b = EDGE_M
    yy, xx = to_geographic([[-b, -b, 0], [b, -b, 0], [b, b, 0], [-b, b, 0], [-b, -b, 0]]).T
    map_ax.plot(xx, yy, color="#132f3d", lw=1.5, ls="--", label="10 × 10 km search")
    coarse = output / "grid_500m.csv"
    if coarse.exists():
        grid = pd.read_csv(coarse).dropna(subset=["loss_hz2"])
        if not grid.empty:
            g_lat, g_lon = to_geographic(grid[XYZ]).T
            scores = np.sqrt(grid.loss_hz2)
            artist = map_ax.scatter(g_lon, g_lat, c=scores, cmap="viridis_r", marker="s", s=69,
                                    linewidths=0, alpha=.74, zorder=3)
            cbar = fig.colorbar(artist, ax=map_ax, pad=.025, fraction=.042, shrink=.82)
            cbar.set_label("Best profile RMSE (Hz) · lower is better", fontsize=8)
            cbar.ax.tick_params(labelsize=8)
    station_lat, station_lon = evaluation["reference_lat_lon"]
    map_ax.scatter(station_lon, station_lat, c="#0d3341", edgecolors="white", linewidths=1.4,
                   s=118, marker="*", zorder=5, label="Published station")
    if diagnostic := infer.get("diagnostic_point_lat_lon"):
        a, b = diagnostic
        map_ax.scatter(b, a, c="#e14e46", edgecolors="white", linewidths=1, s=53, marker="D", zorder=5,
                       label="Lowest residual")
    map_ax.set(xlim=BASEMAP_EXTENT[:2], ylim=BASEMAP_EXTENT[2:], xlabel="Longitude (°)", ylabel="Latitude (°)")
    map_ax.set_aspect(1 / np.cos(np.deg2rad(LAT)))
    map_ax.xaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    map_ax.yaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    map_ax.tick_params(labelsize=8)
    map_ax.legend(loc="lower left", fontsize=8, framealpha=.94)

    sr, audio = read_audio(output / "input.wav")
    spec = make_spectrogram(audio, sr, **STFT_SETTINGS)
    stride = max(1, len(spec.t) // 1250)
    shown = audio_ax.pcolormesh(spec.t[::stride] / 60, spec.f, spec.whitened[:, ::stride],
                                vmin=-1, vmax=8, cmap="magma", shading="auto", rasterized=True)
    for i, ridge in enumerate(ridges):
        sec, freq = np.asarray(ridge["trace_points"]).T
        audio_ax.plot(sec / 60, freq, c="#73f5d5", lw=1.4, label="Selected ridge" if not i else None)
    if ridges:
        audio_ax.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(shown, ax=audio_ax, pad=.025, fraction=.042, shrink=.82)
    cbar.set_label("Whitened STFT intensity", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    audio_ax.set(xlim=(0, infer["duration_s"] / 60), ylim=(60, 450),
                 xlabel="Minutes from clip start", ylabel="Frequency (Hz)")
    audio_ax.tick_params(labelsize=8)
    error = evaluation["diagnostic_great_circle_error_m"]
    detail = (f"Lowest-residual point: {error:,.0f} m from published station. Grid colors are fit residuals."
              if error is not None else "No track-scored point was produced.")
    fig.text(.077, .101, f"{infer['ridges_selected']} selected ridge(s). {detail}",
             fontsize=10, color="#263c45")
    fig.text(.077, .061, "Basemap: USGS The National Map · Audio: Eclipse Soundscapes, CC BY 4.0 · Tracks: adsb.lol, ODbL 1.0",
             fontsize=8, color="#4a5d66")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=145, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--basemap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args.run, args.basemap, args.output)


if __name__ == "__main__":
    main()
