"""Visualize sampled evidence on the full synthetic box; never a probability map."""
from pathlib import Path
import numpy as np
import pandas as pd


def plot_benchmark(output, run_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    output = Path(output)
    all_cases = run_summary["cases_completed"]
    with_grid = [s for s in all_cases if (output/s/"results"/"grid_500m.csv").exists()]
    # Main image shows held-out positives and track-matching controls;
    # pilot detailed results are still preserved in the CSV and JSON.
    plotted = [s for s in with_grid if all_cases[s]["truth"]["split"] != "pilot"] or with_grid
    if not plotted:
        return
    dark = "#101925"
    light = "#eff6fe"
    muted = "#a7bbcd"
    fig,axes = plt.subplots(2,3,figsize=(13.8,10.5),facecolor=dark)
    axs = tuple(axes.ravel())
    norm = Normalize(vmin=0,vmax=4,clip=True)
    colors = plt.colormaps["viridis_r"].copy()
    colors.set_bad(color="#445267")
    titles = {"holdout-nw":"Held-out northwest", "holdout-se":"Held-out southeast",
              "holdout-ne":"Held-out northeast", "holdout-sw":"Held-out southwest",
              "west":"West edge", "southeast":"Southeast corner", "northeast":"Northeast zone",
              "interior":"Interior", "ambiguous":"Spatial ambiguity control", "wrong-tracks":"Wrong-track control"}
    for ax,slug in zip(axs,plotted):
        report = all_cases[slug]
        folder = output/slug/"results"
        coarse = pd.read_csv(folder/"grid_500m.csv")
        table = coarse.pivot(index="y_m",columns="x_m",values="loss_hz2")
        best = report.get("best_fit_diagnostic",{}).get("loss_hz2",0.)
        ax.set_facecolor(dark)
        ax.pcolormesh(table.columns/1000,table.index/1000,np.ma.masked_invalid(table.values-best),
                      shading="nearest",norm=norm,cmap=colors)
        fine = folder/"grid_25m.csv"
        if fine.exists():
            fine_data = pd.read_csv(fine)
            near = fine_data[fine_data.loss_hz2 <= best+report["settings"]["near_best_loss_hz2"]]
            ax.scatter(near.x_m/1000,near.y_m/1000,c="#c1fff1",s=10,alpha=.85,
                       linewidths=0,zorder=3)
        xp = report.get("estimated_xyz_m")
        if xp is not None:
            ax.scatter(xp[0]/1000,xp[1]/1000,marker="+",c="#fff",s=170,linewidths=2,zorder=8)
        xp = report["truth"]["stationary_virtual_receiver_xyz_m"]
        if xp is not None:
            ax.scatter(xp[0]/1000,xp[1]/1000,marker="*",c="#ffab5a",s=85,edgecolor=dark,linewidths=.65,zorder=9)
        xp = report["truth"].get("equivalent_xyz_m")
        if xp is not None:
            ax.scatter(xp[0]/1000,xp[1]/1000,marker="*",c="#ffab5a",s=85,edgecolor=dark,linewidths=.65,zorder=9)
        err = report["position_error_m"]
        if err is None:
            caption = "abstained"
        elif report["truth"]["expected_outcome"] == "abstain":
            caption = f"false match ({err/1000:.2f} km)"
        else:
            caption = f"{err:.1f} m from truth"
        ax.set_title(f"{titles.get(slug,slug)}  ·  {caption}",color=light,fontsize=12,pad=11,loc="left")
        ax.set(xlim=(-5,5),ylim=(-5,5),aspect="equal",xticks=(-5,0,5),yticks=(-5,0,5),
               xlabel="Local x (km)",ylabel="Local y (km)")
        ax.tick_params(colors=muted)
        ax.xaxis.label.set_color(muted)
        ax.yaxis.label.set_color(muted)
        for spine in ax.spines.values():
            spine.set_color("#7490a3")
    for ax in axs[len(plotted):]:
        ax.axis("off")
    fig.subplots_adjust(left=.058,right=.94,bottom=.13,top=.82,wspace=.25,hspace=.34)
    cbar_ax = fig.add_axes([.941,.235,.014,.46])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=colors),cax=cbar_ax,extend="max")
    cb.set_label("Additional mean squared ridge error (Hz²)",color=muted,fontsize=9,labelpad=8)
    cb.ax.tick_params(colors=muted)
    fig.text(.070,.949,"Aircraft Doppler  /  Wide synthetic benchmark",color=light,fontsize=23,weight="bold")
    fig.text(.072,.904,"10 × 10 km initial grid  •  27 minutes of audio per case  •  ±80 s recording clock uncertainty",
             color="#8cf1db",fontsize=12,weight="bold")
    fig.text(.072,.864,"Coarse colors compare fits within each panel, not probabilities. Gray means no feasible joint score.",
             color=muted,fontsize=10)
    handles = [
        Line2D([],[],marker="*",color="#ffab5a",linestyle="",markersize=12,label="Synthetic truth or identical alternative"),
        Line2D([],[],marker="+",color="#fff",linestyle="",markersize=11,label="Accepted estimate"),
        Line2D([],[],marker=".",color="#c1fff1",linestyle="",markersize=8,label="Near-best sampled fine points"),
    ]
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.067,.062),ncols=3,labelcolor=light,frameon=False,
               columnspacing=2.15,fontsize=10)
    fig.text(.072,.038,"Pilot results and the no-flyovers audio-only control are shown in the report, not in the grid panels.",
             color=muted,fontsize=9)
    fig.savefig(output/"wide_heatmap.png",dpi=155,facecolor=dark)
    plt.close(fig)



def plot_timeline(output, run_summary):
    """Use one actually replayed holdout, with no interpolation of truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .audio import read_audio, make_spectrogram
    from .benchmark import STFT_SETTINGS
    from .physics import forward_curve
    from .io import load_tracks

    output = Path(output)
    selected = next(((s,r) for s,r in run_summary["cases_completed"].items()
                     if "estimated_pair" in r and r["truth"]["split"]=="holdout"),None)
    if selected is None:
        return
    slug, report = selected
    case = output/slug
    sr,audio = read_audio(case/"inputs"/"synthetic.wav")
    spec = make_spectrogram(audio,sr,**STFT_SETTINGS)
    best = pd.read_csv(case/"results"/"best_per_pair.csv").iloc[0]
    tracks = {tr.flight_id:tr for tr in load_tracks(case/"inputs"/"tracks.csv")}
    dark, light, muted = "#101925","#eff6fe","#a7bbcd"
    fig = plt.figure(figsize=(13.8,8.25),facecolor=dark)
    grids = fig.add_gridspec(nrows=2,ncols=2,left=.075,right=.95,bottom=.13,top=.79,
                            hspace=.55,wspace=.27,height_ratios=[1,1.12])
    ax = fig.add_subplot(grids[0,:])
    selected_band = (spec.f >= 65)&(spec.f <= 195)
    ax.pcolormesh(spec.t/60,spec.f[selected_band],spec.whitened[selected_band,:],cmap="magma",
                  shading="nearest", vmin=-2,vmax=17, rasterized=True)
    for peak in report["audio_audit"]["rejected"]:
        ax.axvline(peak["tonal_peak_audio_s"]/60,color="#a7bbcd",linewidth=.65,linestyle=":",alpha=.8)
    for i, event in enumerate(report["audio_only_detection"],1):
        minute = event["initial_ideal_fit"]["delayed_cpa_audio_s"]/60
        ax.axvline(minute,color="#8cf1db",linewidth=1.3)
        ax.text(minute+.15,174,f"Tone {i}",color="#8cf1db",fontsize=9,weight="bold")
    ax.set(title="Full 27-minute audio spectrogram  ·  dotted lines are rejected candidate peaks",
           xlabel="Recording time (min)",ylabel="Frequency (Hz)",xlim=(0,27),ylim=(65,195))
    for i, fid in enumerate(best.flight_pair.split("|"),1):
        bx = fig.add_subplot(grids[1,i-1])
        event = report["audio_only_detection"][i-1]
        seconds = event["tonal_peak_audio_s"]
        use = abs(spec.t-seconds) <= 32
        tr = forward_curve(tracks[fid],report["estimated_xyz_m"])
        zoom_t = tr.arrival_s-report["estimated_clock_offset_s"]
        fundamental = tr.ratio*best[f"event_{i}_source_hz"]
        bx.pcolormesh(spec.t[use]/60,spec.f[selected_band],spec.whitened[selected_band,:][:,use],
                     cmap="magma",shading="nearest",vmin=-2,vmax=17)
        bx.plot(zoom_t/60,fundamental,color="#8cf1db",linewidth=.85,label="Recovered Doppler curve")
        bx.set(xlim=(spec.t[use][0]/60,spec.t[use][-1]/60),ylim=(65,195),
               xlabel="Recording time (min)",ylabel="Frequency (Hz)",
               title=f"Tone {i} details  ·  track {fid}")
        bx.legend(loc="upper right",fontsize=8,labelcolor=light,facecolor=dark,edgecolor=muted)
    for ax in fig.axes:
        ax.tick_params(colors=muted)
        ax.xaxis.label.set_color(muted)
        ax.yaxis.label.set_color(muted)
        ax.title.set(color=light,weight="bold",fontsize=12)
        for spine in ax.spines.values():
            spine.set_color("#7490a3")
    gap = report["audio_event_separation_s"]/60
    fig.text(.075,.933,"Two Doppler signatures across one long recording",fontsize=23,color=light,weight="bold")
    fig.text(.077,.878,f"{slug.upper()}  /  {gap:.1f} minutes between signatures  /  {report['position_error_m']:.1f} m from synthetic truth",
             fontsize=12,color="#8cf1db",weight="bold")
    fig.text(.077,.053,"The waveform, candidate tracks and ground truth are all simulation-only.",
             color=muted,fontsize=10)
    fig.savefig(output/"evidence_timeline.png",dpi=155,facecolor=dark)
    plt.close(fig)
