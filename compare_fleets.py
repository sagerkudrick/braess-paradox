"""
Humans vs central planner, shortcut closed vs open, across fleet sizes.

    python compare_fleets.py
    python compare_fleets.py --agents 20 40 60 --seeds 6
"""
import argparse
from multiprocessing import Pool

import numpy as np

from sim import Sim


def run(args):
    n, fleet, shortcut, seed, steps = args
    sim = Sim(n, fleet, shortcut, seed=seed, layout_kw={"shortcut_w": 1})
    r = sim.run(steps, 300)
    avoided = np.mean([x[1] for x in sim.avoid_log]) if sim.avoid_log else float("nan")
    return n, fleet, shortcut, r["throughput_per_1000"], avoided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()
    fleets = ("human", "coordinated")
    jobs = [(n, f, s, k, a.steps) for n in a.agents for f in fleets for s in (False, True)
            for k in range(a.seeds)]
    with Pool() as pool:
        rows = pool.map(run, jobs, chunksize=1)

    print(f"{'fleet':<12}{'agents':>7}{'closed':>9}{'open':>9}{'change':>9}   planner avoided shortcut")
    for n in a.agents:
        for f in fleets:
            c = np.mean([r[3] for r in rows if r[:3] == (n, f, False)])
            o = np.mean([r[3] for r in rows if r[:3] == (n, f, True)])
            av = np.nanmean([r[4] for r in rows if r[:3] == (n, f, True)]) if f == "coordinated" else None
            note = "" if av is None else f"{100 * av:.0f}% of decisions"
            print(f"{f:<12}{n:>7}{c:>9.0f}{o:>9.0f}{100 * (o - c) / c:>+8.0f}%   {note}")


if __name__ == "__main__":
    main()
