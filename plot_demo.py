"""Render a synthetic replay result. Truth is used here for the plot only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from aircraft_doppler.audio import make_spectrogram, read_audio
from aircraft_doppler.benchmark import STFT_SETTINGS
from aircraft_doppler.io import load_tracks
from aircraft_doppler.physics import forward_curve


def render(result, destination):
    fixture = Path(__file__).resolve().parent/"example"
    # Inference is already complete. Loading the answer cannot change the fit.
    report_path = result/"inference.json"
    report = json.loads((report_path if report_path.is_file() else result/"summary.json").read_text())
    answer = json.loads((fixture/"truth.json").read_text())
    x, y, z = report["estimated_xyz_m"]
    ax, ay, az = answer["stationary_virtual_receiver_xyz_m"]
    dist = float(np.linalg.norm([x-ax, y-ay, z-az]))
    coarse = pd.read_csv(result/"grid_500m.csv").pivot(index="y_m",columns="x_m",values="loss_hz2")
    finer = pd.read_csv(result/"grid_25m.csv")
    loss = report["best_fit_diagnostic"]["loss_hz2"]
    best = pd.read_csv(result/"best_per_pair.csv").iloc[0]
    events = json.loads((result/"detected_events.json").read_text())
    rate, audio = read_audio(fixture/"synthetic.wav")
    spec = make_spectrogram(audio,rate,**STFT_SETTINGS)
    band = (spec.f>=65)&(spec.f<=195)
    tracks = {t.flight_id:t for t in load_tracks(fixture/"tracks.csv")}
    background, ink, accent, quiet = "#101925","#edf5fb","#8cf1db","#a6bacb"
    fig = plt.figure(figsize=(14,7.6),facecolor=background)
    g = fig.add_gridspec(2,3,left=.073,right=.965,bottom=.145,top=.763,
                         width_ratios=[1.37,.9,.9],hspace=.62,wspace=.34)
    m = fig.add_subplot(g[:,0])
    tone = fig.add_subplot(g[0,1:])
    panels = [fig.add_subplot(g[1,1]),fig.add_subplot(g[1,2])]
    m.pcolormesh(coarse.columns/1000,coarse.index/1000,coarse.values-loss,
                 cmap="viridis_r",norm=Normalize(vmin=0,vmax=4,clip=True),shading="nearest")
    near = finer[finer.loss_hz2 <= loss+report["settings"]["near_best_loss_hz2"]]
    m.scatter(near.x_m/1000,near.y_m/1000,color=accent,s=9,alpha=.9,linewidths=0)
    m.scatter([ax/1000],[ay/1000],color="#ffb371",marker="*",s=165,label="Synthetic truth")
    m.scatter([x/1000],[y/1000],color="#fff",marker="+",s=75,label="Recovered",linewidths=1.6)
    m.set(xlim=(-5,5),ylim=(-5,5),aspect="equal",xticks=(-5,0,5),yticks=(-5,0,5),
          xlabel="Local x (km)",ylabel="Local y (km)",title="10 × 10 km search")
    m.legend(loc="upper left",fontsize=8,labelcolor=ink,facecolor=background,edgecolor=quiet)
    tone.pcolormesh(spec.t/60,spec.f[band],spec.whitened[band],cmap="magma",shading="nearest",
                    vmin=-2,vmax=17,rasterized=True)
    for i, event in enumerate(events,1):
        time = event["cpa_audio_s"]/60
        tone.axvline(time,color=accent,linewidth=1)
        tone.text(time+.18,169,f"Tone {i}",color=accent,fontsize=8,weight="bold")
    tone.set(xlim=(0,27),ylim=(65,195),xlabel="Time (min)",ylabel="Hz",
             title="27-minute recording")
    for i,(p,event,track_id) in enumerate(zip(panels,events,report["estimated_pair"].split("|")),1):
        lo,hi = event["window_s"]
        trace = np.asarray(event["trace_points"])
        use = (lo <= spec.t)&(spec.t <= hi)
        p.pcolormesh(spec.t[use],spec.f[band],spec.whitened[band][:,use],cmap="magma",shading="nearest",
                     vmin=-2,vmax=17)
        curve=forward_curve(tracks[track_id],[x,y,z])
        p.plot(curve.arrival_s-report["estimated_clock_offset_s"],curve.ratio*best[f"event_{i}_source_hz"],
               color=accent,linewidth=.9)
        p.set(xlim=(lo,hi),ylim=(min(trace[:,1])-7,max(trace[:,1])+7),
              xlabel="Time (s)",ylabel="Hz",title=f"Tone {i} · {track_id}")
    for p in [m,tone,*panels]:
        p.set_facecolor(background)
        p.tick_params(colors=quiet,labelsize=8)
        p.title.set(color=ink,weight="bold",fontsize=11)
        p.xaxis.label.set(color=quiet,fontsize=9)
        p.yaxis.label.set(color=quiet,fontsize=9)
        for spine in p.spines.values():
            spine.set_color("#526981")
    fig.text(.074,.922,"Localizing a virtual microphone from Doppler",color=ink,fontsize=23,weight="bold")
    fig.text(.075,.862,f"SYNTHETIC EXAMPLE   /   {dist:.1f} m error   /   {report['audio_event_separation_s']/60:.1f} min between tones",
             color=accent,fontsize=12,weight="bold")
    fig.text(.077,.063,"Map: relative fit on sampled cells, not probability. Curves: guarded fit, conditional on the supplied track catalogue.",
             color=quiet,fontsize=9)
    fig.savefig(destination,dpi=145,facecolor=background)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",type=Path,default=Path("evidence"))
    parser.add_argument("--output",type=Path,required=True)
    options=parser.parse_args()
    render(options.result,options.output)
