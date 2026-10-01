"""
Sweep fleet type x fleet size x shortcut on/off, several seeds each, in parallel.

    python experiment.py                      # default sweep
    python experiment.py --shortcut-w 2       # wide shortcut instead of narrow
    python experiment.py --agents 20 40 60 80 100 --seeds 8

Writes results.csv and braess.png.
"""
import argparse
import csv
import itertools
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from sim import FLEETS, Sim


def run_one(args):
    fleet, n, shortcut, seed, steps, shortcut_w = args
    r = Sim(n, fleet, shortcut, seed=seed, layout_kw={"shortcut_w": shortcut_w}).run(steps)
    return {"fleet": fleet, "agents": n, "shortcut": shortcut, "seed": seed, **r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--shortcut-w", type=int, default=1)
    ap.add_argument("--fleets", nargs="+", default=list(FLEETS))
    a = ap.parse_args()

    jobs = [(f, n, s, k, a.steps, a.shortcut_w)
            for f, n, s, k in itertools.product(a.fleets, a.agents, (False, True), range(a.seeds))]
    print(f"running {len(jobs)} simulations...")
    with Pool() as pool:
        rows = pool.map(run_one, jobs, chunksize=1)

    with open("results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def agg(f, n, s, key="throughput_per_1000"):
        v = [r[key] for r in rows if r["fleet"] == f and r["agents"] == n and r["shortcut"] == s]
        return np.mean(v), np.std(v) / np.sqrt(len(v))

    colors = {"human": "#d9534f", "amr": "#f0ad4e", "coordinated": "#2b7bba"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    print(f"\n{'fleet':<12}{'agents':>7}{'off':>9}{'on':>9}{'change':>9}")
    for f in a.fleets:
        off = np.array([agg(f, n, False) for n in a.agents])
        on = np.array([agg(f, n, True) for n in a.agents])
        c = colors.get(f)
        ax1.errorbar(a.agents, off[:, 0], off[:, 1], color=c, ls="--", marker="o", label=f"{f} - no shortcut")
        ax1.errorbar(a.agents, on[:, 0], on[:, 1], color=c, ls="-", marker="s", label=f"{f} - shortcut")
        pct = 100 * (on[:, 0] - off[:, 0]) / off[:, 0]
        ax2.plot(a.agents, pct, color=c, marker="o", label=f)
        for n, o1, o2, p in zip(a.agents, off[:, 0], on[:, 0], pct):
            print(f"{f:<12}{n:>7}{o1:>9.0f}{o2:>9.0f}{p:>8.1f}%")

    ax1.set(xlabel="fleet size", ylabel="orders completed / 1000 steps", title="Throughput")
    ax1.legend(fontsize=8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set(xlabel="fleet size", ylabel="% throughput change from opening shortcut",
            title="Braess indicator (below 0 = paradox)")
    ax2.legend()
    fig.suptitle(f"Warehouse Braess experiment (shortcut width = {a.shortcut_w}, {a.seeds} seeds)")
    fig.tight_layout()
    fig.savefig("braess.png", dpi=130)
    print("\nwrote results.csv and braess.png")


if __name__ == "__main__":
    main()
