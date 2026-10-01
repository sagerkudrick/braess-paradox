"""
Live side-by-side view: same fleet, same orders, shortcut CLOSED (left) vs OPEN (right).

    python visualize.py --fleet human --agents 80
    python visualize.py --fleet coordinated --agents 80
    python visualize.py --mix human=0.5 coordinated=0.5 --agents 80
    python visualize.py --fleet human --agents 80 --save demo.gif --frames 400
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from sim import FLEETS, Sim

COLORS = {"human": "#d9534f", "amr": "#f0ad4e", "coordinated": "#2b7bba"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", choices=FLEETS, default="human")
    ap.add_argument("--mix", nargs="*", help="e.g. human=0.5 coordinated=0.5")
    ap.add_argument("--agents", type=int, default=80)
    ap.add_argument("--shortcut-w", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", help="save to .gif instead of showing a window")
    ap.add_argument("--frames", type=int, default=600)
    a = ap.parse_args()
    mix = {k: float(v) for k, v in (m.split("=") for m in a.mix)} if a.mix else None

    sims = [Sim(a.agents, a.fleet, sc, mix=mix, seed=a.seed,
                layout_kw={"shortcut_w": a.shortcut_w}) for sc in (False, True)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    cmap = ListedColormap(["#3b3b3b", "#f4f1ea", "#7cc47c", "#8a6d4b"])   # rack, floor, dock, wall
    scat, texts, heat = [], [], []
    for ax, sim, title in zip(axes, sims, ("Shortcut CLOSED", "Shortcut OPEN")):
        g = sim.lay.grid.copy()
        ax.imshow(g, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        heat.append(ax.imshow(np.zeros(g.shape), cmap="Reds", alpha=0.0, vmin=0, vmax=1,
                              interpolation="nearest"))
        if sim.lay.shortcut_cells and title.endswith("OPEN"):
            r, c = np.unravel_index(sim.lay.shortcut_cells, g.shape)
            ax.scatter(c, r, marker="s", s=60, facecolors="none", edgecolors="#9b59b6", lw=0.6)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        scat.append(ax.scatter([], [], s=40, edgecolors="k", linewidths=0.4))
        texts.append(ax.text(0.01, -0.06, "", transform=ax.transAxes, fontsize=10, va="top"))

    who = ", ".join(f"{k} {v:.0%}" for k, v in mix.items()) if mix else a.fleet
    fig.suptitle(f"Warehouse twin - {a.agents} agents ({who}), shortcut width {a.shortcut_w}")

    def update(frame):
        for sim, sc, tx, hm in zip(sims, scat, texts, heat):
            sim.step()
            W = sim.W
            pos = np.array([divmod(ag.pos, W) for ag in sim.agents])
            sc.set_offsets(pos[:, ::-1])
            cols = []
            for ag in sim.agents:
                c = COLORS[ag.kind]
                cols.append("#222222" if ag.phase in ("picking", "packing") else c)
            sc.set_facecolors(cols)
            if sim.t % 20 == 0 and sim.visits.max() > 0:
                v = sim.visits.reshape(sim.lay.shape)
                hm.set_data(v / v.max())
                hm.set_alpha(0.35)
            blocked = sum(ag.blocked > 0 for ag in sim.agents)
            tx.set_text(f"t={sim.t}   orders done={sim.completed}   blocked now={blocked}")
        return scat + texts + heat

    anim = FuncAnimation(fig, update, frames=a.frames, interval=30, blit=False)
    if a.save:
        anim.save(a.save, fps=25)
        print("saved", a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
