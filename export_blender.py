"""
Run the simulations to show in Blender and save every agent's position at every step
for blender_scene.py to animate.

    python export_blender.py --compare fleets                 # humans vs AVs, shortcut open
    python export_blender.py --compare fleets --shortcut closed
    python export_blender.py --compare all                    # 2x2: humans/AVs x closed/open
    python export_blender.py --compare shortcut --fleet human # one fleet, closed vs open

All panels use the same agents and the same order list, so the only differences
are the ones being compared.
"""
import argparse
import json

from sim import FLEETS, Sim

NAMES = {"human": "Human drivers (selfish)",
         "coordinated": "Autonomous (central planner)",
         "amr": "Naive robots"}


def record(n, fleet, shortcut, seed, shortcut_w, warmup, steps):
    sim = Sim(n, fleet, shortcut, seed=seed, layout_kw={"shortcut_w": shortcut_w})
    for _ in range(warmup):
        sim.step()
    done0 = sim.completed
    frames, orders, waiting = [], [], []
    for _ in range(steps):
        sim.step()
        frames.append([ag.pos for ag in sim.agents])
        orders.append(sim.completed - done0)
        waiting.append(sum(ag.blocked > 0 for ag in sim.agents))
    return sim, frames, orders, waiting


def make_data(compare="all", fleet="human", shortcut="open", agents=50, shortcut_w=1,
              seed=0, warmup=300, steps=500, log=print):
    # (fleet, shortcut_open, row, col)
    if compare == "fleets":
        panels = [("human", shortcut == "open", 0, 0), ("coordinated", shortcut == "open", 0, 1)]
    elif compare == "shortcut":
        panels = [(fleet, False, 0, 0), (fleet, True, 0, 1)]
    else:
        panels = [("human", False, 0, 0), ("human", True, 0, 1),
                  ("coordinated", False, 1, 0), ("coordinated", True, 1, 1)]

    scenes = []
    for fl, sc_open, row, col in panels:
        sim, frames, orders, waiting = record(agents, fl, sc_open, seed, shortcut_w, warmup, steps)
        label = f"{NAMES[fl]} - shortcut {'OPEN' if sc_open else 'CLOSED'}"
        scenes.append({
            "label": label,
            "fleet": fl,
            "shortcut": sc_open,
            "row": row,
            "col": col,
            "grid": sim.lay.grid.tolist(),
            "shortcut_cells": sim.lay.shortcut_cells,
            "kinds": [ag.kind for ag in sim.agents],
            "positions": frames,          # [step][agent] -> flat cell index (row * W + col)
            "orders": orders,             # [step] -> orders completed since recording began
            "waiting": waiting,           # [step] -> vehicles currently stuck in traffic
        })
        log(f"{label:<55} {orders[-1]:>4} orders in {steps} steps")
    return {"steps": steps, "agents": agents, "compare": compare, "scenes": scenes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", choices=("fleets", "shortcut", "all"), default="all")
    ap.add_argument("--fleet", choices=FLEETS, default="human", help="for --compare shortcut")
    ap.add_argument("--shortcut", choices=("open", "closed"), default="open", help="for --compare fleets")
    ap.add_argument("--agents", type=int, default=50)
    ap.add_argument("--shortcut-w", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=300, help="steps to run before recording")
    ap.add_argument("--steps", type=int, default=500, help="steps to record")
    ap.add_argument("--out", default="trajectories.json")
    a = ap.parse_args()
    data = make_data(a.compare, a.fleet, a.shortcut, a.agents, a.shortcut_w, a.seed, a.warmup, a.steps)
    with open(a.out, "w") as fh:
        json.dump(data, fh)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
