"""Pictures: a snapshot of the world and an end-of-run report. Uses matplotlib only (no display needed)."""
import csv
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap

from .genes import animal_phenotype, plant_phenotype
from .world import INTENTS

BG, FG, GRID = "#090d12", "#d7e3ee", "#223142"
_tox = LinearSegmentedColormap.from_list("tox", ["#3fbf8f", "#c8d86a", "#c04bd8"])


def snapshot(world, cortex=None, path=None, size=6.0):
    """Draw the world. Plants are coloured by toxin (green = sweet, violet = bitter). Animals take their lineage colour."""
    cfg = world.cfg
    fig, ax = plt.subplots(figsize=(size, size), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    # the warm band as a faint background
    xs = [i / 200 * cfg.size for i in range(201)]
    import torch
    t = world.temperature(torch.tensor(xs)).cpu().numpy()
    ax.imshow(t[None, :], extent=[0, cfg.size, 0, cfg.size], aspect="auto", cmap="magma", alpha=0.22, vmin=0, vmax=1.3)
    p, a = world.p, world.a
    pp = plant_phenotype(p["genes"])
    pos = p["pos"].cpu()
    ax.scatter(pos[:, 0], pos[:, 1], s=(p["biomass"].cpu() * 26 + 3), c=pp["toxin"].cpu(), cmap=_tox, vmin=0, vmax=1, marker="o", linewidths=0, alpha=0.9)
    ph = animal_phenotype(a["genes"])
    ap = a["pos"].cpu()
    hue = a["genes"][:, -1].cpu().numpy()
    cols = [hsv_to_rgb((h, 0.75, 0.95)) for h in hue]
    hunter = (ph["diet"] > 0.45).cpu().numpy()
    sz = (ph["size"].cpu().numpy() * 34)
    for mk, m in (("o", ~hunter), ("X", hunter)):
        if m.any():
            ax.scatter(ap[m, 0], ap[m, 1], s=sz[m], c=[c for c, k in zip(cols, m) if k], marker=mk, edgecolors="#ffffff", linewidths=0.4)
    if cortex is not None:
        ids = a["id"].cpu().tolist()
        for aid, mem in cortex.memory.items():
            if aid in ids and bool(a["llm"][ids.index(aid)]):
                i = ids.index(aid)
                ax.scatter(ap[i, 0], ap[i, 1], s=170, facecolors="none", edgecolors="#ffb347", linewidths=1.4)
                intent = a["intent"][i].item()
                ax.annotate(f"{mem['name']} ({INTENTS[intent] if intent >= 0 else '-'})", (ap[i, 0], ap[i, 1]), color="#ffb347",
                            fontsize=7, xytext=(6, 6), textcoords="offset points")
    ax.set_xlim(0, cfg.size); ax.set_ylim(0, cfg.size)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(f"tick {world.tick}   animals {ap.shape[0]}   plants {pos.shape[0]}", color=FG, fontsize=9, loc="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, facecolor=BG)
        plt.close(fig)
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf


def report(stats_csv, out_png):
    rows = list(csv.DictReader(open(stats_csv)))
    if len(rows) < 2:
        return
    col = lambda k: [float(r[k]) for r in rows]
    tick = col("tick")
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), dpi=100)
    fig.patch.set_facecolor(BG)
    panels = [
        ("Populations", [("animals", "#ffb347"), ("plants", "#5eead4"), ("hunters", "#ff5d73")]),
        ("Mean lifespan and vision range", [("mean_lifespan", "#5eead4"), ("a_vision", "#e5c24a")]),
        ("Coevolution: plant toxin vs animal resistance", [("p_toxin", "#c04bd8"), ("a_resist", "#ffb347")]),
        ("Animal diet, aggression, curiosity", [("a_diet", "#ff5d73"), ("a_aggr", "#ffb347"), ("a_cur", "#5eead4")]),
        ("Animal body", [("a_size", "#ffb347"), ("a_speed", "#5eead4")]),
        ("Plant traits", [("p_fecund", "#e5c24a"), ("p_warm", "#e879f9"), ("p_toxin", "#c04bd8")]),
    ]
    for ax, (title, series) in zip(axs.flat, panels):
        ax.set_facecolor(BG)
        for key, color in series:
            ax.plot(tick, col(key), color=color, lw=1.8, label=key)
        ax.set_title(title, color=FG, fontsize=10, loc="left")
        ax.tick_params(colors="#8195a8", labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.grid(color=GRID, lw=0.5, alpha=0.6)
        ax.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=BG)
    plt.close(fig)
