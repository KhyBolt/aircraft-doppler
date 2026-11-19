"""Offline plot of *sampled* candidate scores, not a probability surface."""
import numpy as np


def plot_score_heatmap(coarse, fine, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    theme = {"paper":"#101827", "ink":"#eff4fb", "muted":"#a9b7cd", "off":"#2e3b4f", "missing":"#657189"}
    fig = plt.figure(figsize=(12.5, 7.7), dpi=180, facecolor=theme["paper"])
    grid = fig.add_gridspec(1, 3, left=0.080, right=0.93, bottom=0.23, top=0.78,
                           width_ratios=[1, 1, .055], wspace=0.32)
    axes = [fig.add_subplot(grid[i]) for i in (0, 1)]
    color_ax = fig.add_subplot(grid[2])
    colors = plt.colormaps["viridis"].copy()
    colors.set_bad(alpha=0)
    reference = report["best_score_diagnostic"]
    norm = Normalize(vmin=-0.5, vmax=0, clip=True)
    spans = {"coarse":200, "fine":50}
    for ax, table, label in zip(axes, (coarse, fine), ("coarse", "fine")):
        spacing = spans[label]
        x = np.arange(table.x_m.min(), table.x_m.max()+spacing*.5, spacing)
        y = np.arange(table.y_m.min(), table.y_m.max()+spacing*.5, spacing)
        surface = table.pivot(index="y_m", columns="x_m", values="profile_score").reindex(index=y, columns=x)
        tried = table.assign(evaluated=1).pivot(index="y_m", columns="x_m", values="evaluated").reindex(index=y, columns=x)
        ax.set_facecolor(theme["off"])
        ax.pcolormesh(x, y, np.ma.masked_invalid(tried.to_numpy()), shading="nearest",
                      cmap=ListedColormap([theme["missing"]]), norm=Normalize(vmin=0, vmax=1))
        artist = ax.pcolormesh(x, y, np.ma.masked_invalid(surface.to_numpy(float)-reference),
                               shading="nearest", cmap=colors, norm=norm, edgecolors="#10182748", linewidths=.4)
        supported = table[table.profile_score >= reference-report["decision_thresholds"]["support_score_loss"]]
        ax.scatter(supported.x_m, supported.y_m, s=8, color="white", alpha=.72, linewidths=0, zorder=4)
        if "synthetic_truth_xyz_m" in report:
            ax.scatter(*report["synthetic_truth_xyz_m"][:2], marker="*", s=170, color="#ffb453",
                       edgecolor="#0b1322", linewidth=1.0, zorder=7)
        if "estimated_xyz_m" in report:
            ax.scatter(*report["estimated_xyz_m"][:2], marker="+", s=125, color="#fff", linewidth=1.8, zorder=8)
        ax.set(title=f"{'Coarse survey' if label == 'coarse' else 'Fine refinement'}  ·  {spacing} m cells",
               xlabel="Local x (m)", ylabel="Local y (m)", aspect="equal")
        for spine in ax.spines.values():
            spine.set_color("#52617a")
        ax.tick_params(colors=theme["muted"], labelsize=9)
        ax.xaxis.label.set_color(theme["muted"])
        ax.yaxis.label.set_color(theme["muted"])
        ax.title.set(color=theme["ink"], size=12, weight="bold", position=(.5, 1.025))

    cbar = fig.colorbar(artist, cax=color_ax, extend="min")
    cbar.set_label("Score relative to best (higher is better)", color=theme["muted"], labelpad=12)
    cbar.outline.set_edgecolor("#52617a")
    cbar.ax.tick_params(colors=theme["muted"], labelsize=9)
    fig.text(.079,.925,"Aircraft Doppler score heatmap",size=23,weight="bold",color=theme["ink"])
    fig.text(.081,.873,"SYNTHETIC OFFLINE BENCHMARK  /  TWO AUTO-DETECTED FLYOVERS",color="#90ddcf",size=10,weight="bold")
    if "position_error_m" in report:
        summary = f"Recovered point is {report['position_error_m']:.1f} m from synthetic truth."
    else:
        summary = "The benchmark abstained from a point estimate."
    fig.text(.081,.825,summary,color=theme["muted"],size=11)
    handles = []
    if "synthetic_truth_xyz_m" in report:
        handles.append(Line2D([],[],marker="*",color="#ffb453",linestyle="",markersize=12,label="Synthetic truth"))
    if "estimated_xyz_m" in report:
        handles.append(Line2D([],[],marker="+",color="#fff",linestyle="",markersize=9,label="Recovered estimate"))
    handles.extend([Line2D([],[],marker=".",color="#fff",linestyle="",markersize=6,label="Near-best grid point"),
                    Patch(color=theme["missing"],label="Tested, no feasible fit"),
                    Patch(color=theme["off"],label="Not sampled")])
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.074,.128), ncols=len(handles),
               labelcolor=theme["ink"], frameon=False, borderpad=0, columnspacing=1.3, fontsize=9)
    fig.text(.081,.069,"Scores profile over track pair, one shared clock offset, and each tone's frequency. "
             "Colors are not probabilities.",color=theme["muted"],size=9)
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)
