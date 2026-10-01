"""
Warehouse digital-twin simulator for studying Braess's paradox.

A grid warehouse with rack blocks, 2-wide aisles, top/bottom cross-aisles and
an optional middle cross-aisle (the "shortcut"). Agents repeatedly:
    walk to a pick location -> pick (dwell) -> walk to a pack station -> pack (dwell)

Every agent moves at the same speed (1 cell/step) and each cell holds one agent,
so the ONLY thing that differs between fleet types is how they choose routes:

    human        SELFISH (user equilibrium): each person picks the route that
                 minimizes THEIR OWN expected travel time, using congestion they've
                 learned from experience, and periodically re-chooses.
    amr          naive baseline: pure shortest distance, ignores congestion,
                 replans only when physically blocked
    coordinated  CENTRAL PLANNER: plans every vehicle in space AND time. It books
                 which cell each vehicle will occupy at each future step
                 (a reservation table) and plans each route around everyone
                 else's bookings, including deliberate waits, so planned
                 vehicles never meet head-on or gridlock. Every few steps it
                 re-plans the whole fleet. Every 100 steps it also runs a
                 what-if simulation of the next 80 steps with and without the
                 shortcut and tells its vehicles to use or avoid it.

Humans use a learned congestion map T[c] (expected steps to cross cell c) and
route to minimize their own expected time, i.e. the Wardrop user equilibrium.

Braess's paradox shows up if opening the shortcut LOWERS throughput.
"""
from __future__ import annotations

import copy
import heapq
from collections import deque
import random
from dataclasses import dataclass, field

import numpy as np

WALL, FREE, PACK, BARRIER = 0, 1, 2, 3        # WALL = rack, BARRIER = building wall
WALKABLE = (FREE, PACK)
FLEETS = ("human", "amr", "coordinated")


# ----------------------------------------------------------------------------- layout
@dataclass
class Layout:
    grid: np.ndarray            # H x W, values WALL / FREE / PACK / BARRIER
    pick_cells: list[int]       # flat indices of aisle cells facing a rack
    pack_cells: list[int]       # loading-dock stops in the hallway
    spawn_cells: list[int]
    shortcut_cells: list[int]
    neighbors: list[list[int]] = field(default_factory=list)
    narrow: set = field(default_factory=set)

    @property
    def shape(self):
        return self.grid.shape


def build_layout(n_aisles=6, rack_len=18, aisle_w=2, rack_w=2, cross_w=2,
                 shortcut=False, n_packs=7, shortcut_w=2, hall_w=3, one_way=True) -> Layout:
    """
    Columns, left to right:
        hallway (hall_w)  | wall with doorways | racking area
    Loading docks sit along the far-left wall of the hallway, spread over its length.
    Doorways into the racking area line up with the top and bottom cross-aisles;
    when the shortcut is open, the middle cross-aisle also gets a doorway.

    one_way: lane discipline, as in real warehouses. Every 2-wide aisle and cross-aisle
    has one lane per direction (changing lanes is always allowed), and the dock hallway
    is one-way: in at the top doorway, past the docks, out at the bottom. Vehicles never
    meet head-on there. A 1-wide shortcut stays two-way.
    """
    x0 = hall_w + 1                                # first column of the racking area
    W = x0 + n_aisles * aisle_w + (n_aisles - 1) * rack_w
    rack_top = cross_w
    rack_bot = rack_top + rack_len               # exclusive
    H = rack_bot + cross_w
    grid = np.full((H, W), FREE, dtype=np.int8)

    rack_cols = []
    x = x0 + aisle_w
    for _ in range(n_aisles - 1):
        rack_cols.extend(range(x, x + rack_w))
        x += rack_w + aisle_w
    mid0 = rack_top + rack_len // 2 - shortcut_w // 2
    mid_rows = set(range(mid0, mid0 + shortcut_w))

    # hallway wall: doorways at the top and bottom cross-aisles (+ middle via the shortcut)
    wall_c = hall_w
    shortcut_cells = []
    for r in range(rack_top, rack_bot):
        if r in mid_rows:
            shortcut_cells.append(r * W + wall_c)
            if shortcut:
                continue
        grid[r, wall_c] = BARRIER

    for r in range(rack_top, rack_bot):
        for c in rack_cols:
            if r in mid_rows:
                shortcut_cells.append(r * W + c)
                if shortcut:
                    continue
            grid[r, c] = WALL

    # loading docks along the left wall of the hallway
    dock_rows = np.linspace(1, H - 2, n_packs).round().astype(int)
    packs = []
    for r in dock_rows:
        grid[r, 0] = PACK
        packs.append(int(r * W))

    # pick faces: aisle cells horizontally adjacent to a rack that exists in BOTH configs
    pick = []
    rackset = set(rack_cols)
    for r in range(rack_top, rack_bot):
        if r in mid_rows:
            continue
        for c in range(x0, W):
            if c in rackset:
                continue
            if (c - 1 in rackset) or (c + 1 in rackset):
                pick.append(r * W + c)

    sc = set(shortcut_cells)
    spawn = [i for i in range(H * W) if grid.flat[i] == FREE and i not in sc]

    lay = Layout(grid, pick, packs, spawn, shortcut_cells)

    left_lanes = {x0 + k * (aisle_w + rack_w) for k in range(n_aisles)}
    right_lanes = {c + aisle_w - 1 for c in left_lanes}
    cross_groups = [list(range(0, cross_w)), list(range(rack_bot, H))]
    if shortcut and shortcut_w >= 2:
        cross_groups.append(sorted(mid_rows))

    def in_band(r):                                 # rows where aisle/hallway lanes are one-way
        return rack_top <= r < rack_bot

    def allowed(r, c, dr, dc):
        if not one_way:
            return True
        if dr:                                      # moving along a vertical aisle
            if not in_band(r):
                return True
            if c < hall_w or c in left_lanes:       # hallway is a one-way loop past the docks
                return dr > 0
            if c in right_lanes:
                return dr < 0
            return True
        for grp in cross_groups:                    # moving along a cross-aisle
            if r in grp:
                if r == grp[0]:
                    return dc < 0
                if r == grp[-1]:
                    return dc > 0
        return True

    flat = grid.ravel()
    nb = [[] for _ in range(H * W)]
    for i in range(H * W):
        if flat[i] not in WALKABLE:
            continue
        r, c = divmod(i, W)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and flat[rr * W + cc] in WALKABLE and allowed(r, c, dr, dc):
                nb[i].append(rr * W + cc)
    lay.neighbors = nb
    # corridor cells (exactly two opposite neighbours) are too narrow to pass in
    lay.narrow = set()
    for i, ns in enumerate(nb):
        if len(ns) == 2 and abs(ns[0] - ns[1]) in (2, 2 * W):
            lay.narrow.add(i)
    return lay


# ----------------------------------------------------------------------------- routing
def astar(lay: Layout, start: int, goal: int, cost=None, blocked=frozenset()):
    """A* on the 4-connected grid. cost: per-cell entry cost (>=1) or None for unit cost."""
    if start == goal:
        return []
    W = lay.shape[1]
    gr, gc = divmod(goal, W)

    def h(i):
        r, c = divmod(i, W)
        return abs(r - gr) + abs(c - gc)

    g = {start: 0.0}
    came = {}
    openq = [(h(start), 0.0, start)]
    nbrs = lay.neighbors
    while openq:
        _, gcur, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.pop()                      # drop start
            return path[::-1]
        if gcur > g[cur]:
            continue
        for n in nbrs[cur]:
            if n in blocked and n != goal:
                continue
            ng = gcur + (1.0 if cost is None else cost[n])
            if ng < g.get(n, 1e18):
                g[n] = ng
                came[n] = cur
                heapq.heappush(openq, (ng + h(n), ng, n))
    return None


# ----------------------------------------------------------------------------- agents
@dataclass
class Agent:
    id: int
    kind: str                   # one of FLEETS
    pos: int
    rng: random.Random
    phase: str = "to_pick"      # to_pick, picking, to_pack, packing
    goal: int = -1
    path: list = field(default_factory=list)
    dwell: int = 0
    blocked: int = 0
    order_start: int = 0
    enter_t: int = 0            # when the agent entered its current cell
    caused: int = 0             # delay-steps it has imposed on others in this cell


@dataclass
class Params:
    pick_time: int = 6
    pack_time: int = 4
    alpha: float = 0.05         # learning rate of the congestion map (EWMA)
    reroute_every: int = 10     # steps between humans' route re-choices
    window: int = 20            # planner look-ahead (steps it simulates into the future)
    replan_every: int = 5       # steps between whole-fleet re-plans
    plan_orders: int = 1        # priority orders tried per re-plan; best total time wins
    lookahead_every: int = 100  # steps between the planner's what-if simulations (0 = off)
    lookahead_steps: int = 80   # how far ahead each what-if simulation runs
    patience: int = 2           # steps blocked before replanning around obstacles
    unstick: int = 12           # steps blocked before a random side-step


class Sim:
    def __init__(self, n_agents=30, fleet="human", shortcut=False, mix=None,
                 seed=0, params: Params | None = None, layout_kw=None):
        """
        fleet: 'human' | 'amr' | 'coordinated'
        mix:   optional dict e.g. {'human': 0.5, 'coordinated': 0.5} (overrides fleet)
        """
        self.p = params or Params()
        self.lay = build_layout(shortcut=shortcut, **(layout_kw or {}))
        self.H, self.W = self.lay.shape
        self.t = 0
        self.rng = random.Random(seed)
        self.occ: dict[int, int] = {}
        self.res: dict = {}                          # (cell, time) -> vehicle that booked it
        self.eres: dict = {}                         # (from, to, time) -> vehicle moving along it
        self._need_replan = True
        self.avoid: frozenset = frozenset()          # cells the planner tells its vehicles to avoid
        self.avoid_log: list = []                    # (step, avoiding shortcut?) decisions
        self._dist_cache: dict = {}
        self._rev = [[] for _ in self.lay.neighbors]
        for u, ns in enumerate(self.lay.neighbors):
            for v in ns:
                self._rev[v].append(u)
        self.T = np.ones(self.H * self.W)            # learned traversal time per cell
        self.E = np.zeros(self.H * self.W)           # learned externality per cell
        self.visits = np.zeros(self.H * self.W)
        self.completed = 0
        self.cycle_times: list[int] = []
        self.blocked_steps = 0
        self.move_steps = 0
        self.planned_waits = 0

        kinds = []
        if mix:
            for k, frac in mix.items():
                kinds += [k] * round(frac * n_agents)
            kinds = (kinds + [list(mix)[0]] * n_agents)[:n_agents]
        else:
            kinds = [fleet] * n_agents

        spawn = self.rng.sample(self.lay.spawn_cells, n_agents)
        self.agents = []
        for i, (k, s) in enumerate(zip(kinds, spawn)):
            # each agent gets its own RNG so task sequences are identical across
            # shortcut on/off for the same seed -> fair A/B comparison
            a = Agent(i, k, s, random.Random(seed * 1000 + i))
            self.agents.append(a)
            self.occ[s] = i
            self._new_leg(a)

    # --- planning ----------------------------------------------------------------
    def _cost_for(self, a: Agent):
        if a.kind == "amr":
            return None
        return self.T.copy()                         # human: my own expected travel time

    def _plan(self, a: Agent, avoid_occupied=False):
        if a.kind == "coordinated":
            return self._plan_coord(a)
        blocked = frozenset(c for c in self.occ if c != a.pos) if avoid_occupied else frozenset()
        path = astar(self.lay, a.pos, a.goal, self._cost_for(a), blocked)
        if path is None and avoid_occupied:
            return False
        a.path = path or []
        return True

    # --- central planner: space-time reservations ---------------------------------
    DWELLING = ("picking", "packing")

    def _dist_to(self, goal):
        """Exact driving distance from every cell to goal (respects one-way lanes and
        any cells the planner is currently avoiding)."""
        key = (goal, self.avoid)
        d = self._dist_cache.get(key)
        if d is None:
            inf = 10 ** 9
            avoid = self.avoid
            d = [inf] * len(self.lay.neighbors)
            d[goal] = 0
            q = deque([goal])
            while q:
                u = q.popleft()
                if u in avoid and u != goal:
                    continue                          # can't route through avoided cells
                for v in self._rev[u]:
                    if d[v] == inf:
                        d[v] = d[u] + 1
                        q.append(v)
            self._dist_cache[key] = d
        return d

    def _hold(self, a: Agent):
        return self.p.pick_time if a.phase == "to_pick" else self.p.pack_time

    def _st_search(self, a: Agent, t0: int, res: dict, eres: dict):
        """Space-time A*: the earliest arrival at a.goal that never enters a cell, or
        swaps along an edge, that another vehicle has booked. Waiting is a move.
        Returns (cells for t0+1.., reached_goal, estimated total time)."""
        W = self.p.window
        h = self._dist_to(a.goal)
        nb = self.lay.neighbors
        me, goal, hold = a.id, a.goal, self._hold(a)
        avoid = self.avoid
        parent = {(a.pos, 0): None}
        openq = [(h[a.pos], 0, a.pos)]
        end, reached = None, False
        while openq:
            _, negt, c = heapq.heappop(openq)
            t = -negt
            if c == goal and all(res.get((c, t0 + k), me) == me for k in range(t, t + hold + 1)):
                end, reached = (c, t), True
                break
            if t == W:
                end = (c, t)
                break
            T = t0 + t + 1
            for n in nb[c] + [c]:                     # move to a neighbour, or wait
                if n in avoid and n != c:
                    continue
                if res.get((n, T), me) != me:
                    continue
                if n != c and eres.get((n, c, T), me) != me:
                    continue                          # would swap with an oncoming vehicle
                key = (n, t + 1)
                if key in parent:
                    continue
                parent[key] = (c, t)
                heapq.heappush(openq, (t + 1 + h[n], -(t + 1), n))
        if end is None:                               # boxed in: wait and try again later
            return [a.pos], False, 1 + h[a.pos]
        path, node = [], end
        while parent[node] is not None:
            path.append(node[0])
            node = parent[node]
        path.reverse()
        return path, reached, end[1] + (0 if reached else h[end[0]])

    def _book(self, a: Agent, path, reached, t0, res, eres):
        prev = a.pos
        for i, c in enumerate(path):
            T = t0 + i + 1
            res[(c, T)] = a.id
            if c != prev:
                eres[(prev, c, T)] = a.id
            prev = c
        end_t = t0 + len(path)
        for T in range(end_t + 1, end_t + (self._hold(a) if reached else 2) + 1):
            res.setdefault((prev, T), a.id)           # parked at the goal while working

    def _base_bookings(self, t0):
        """Bookings the planner can't choose: parked vehicles, and a prediction of
        where people (who it doesn't control) will drive next."""
        res, W = {}, self.p.window
        for a in self.agents:
            if a.kind == "coordinated":
                span = a.dwell + 1 if a.phase in self.DWELLING else 2
                for k in range(span):
                    res[(a.pos, t0 + k)] = a.id
            else:
                owner = -1 - a.id
                res[(a.pos, t0)] = owner
                res[(a.pos, t0 + 1)] = owner
                for i, c in enumerate(a.path[:3]):
                    res.setdefault((c, t0 + i + 1), owner)
        return res

    def _replan_fleet(self):
        """Re-plan every autonomous vehicle together. Several priority orders are
        simulated; the one with the lowest total travel time is kept."""
        t0 = self.t - 1
        base = self._base_bookings(t0)
        movers = [a for a in self.agents if a.kind == "coordinated" and a.phase not in self.DWELLING]
        best = None
        for trial in range(max(1, self.p.plan_orders)):
            order = movers[:]
            if trial == 0:                            # nearly-there vehicles first
                order.sort(key=lambda a: self._dist_to(a.goal)[a.pos])
            elif trial == 1:                          # longest trips first
                order.sort(key=lambda a: -self._dist_to(a.goal)[a.pos])
            else:
                self.rng.shuffle(order)
            res, eres, plans, total = dict(base), {}, {}, 0
            for a in order:
                if res.get((a.pos, t0 + 1)) == a.id:  # its own plan decides whether it stays
                    del res[(a.pos, t0 + 1)]
                path, reached, cost = self._st_search(a, t0, res, eres)
                self._book(a, path, reached, t0, res, eres)
                plans[a.id] = path
                total += cost
            if best is None or total < best[0]:
                best = (total, plans, res, eres)
        _, plans, self.res, self.eres = best
        for a in movers:
            a.path = list(plans[a.id])
        self._need_replan = False

    def _lookahead(self):
        """What-if simulation: copy the whole warehouse (same future orders), run it
        ahead with the shortcut allowed and with it avoided, keep whichever policy
        completes more orders."""
        closable = frozenset(c for c in self.lay.shortcut_cells if self.lay.neighbors[c])
        if not closable:
            return
        scores = {}
        for avoid in (frozenset(), closable):
            twin = copy.deepcopy(self)
            twin.p = copy.copy(self.p)
            twin.p.lookahead_every = 0                # no what-ifs inside the what-if
            twin.avoid = avoid
            twin._need_replan = True
            for _ in range(self.p.lookahead_steps):
                twin.step()
            scores[avoid] = (twin.completed, -twin.blocked_steps)
        best = max(scores, key=scores.get)
        if best != self.avoid:
            self.avoid = best
            self._need_replan = True
        self.avoid_log.append((self.t, bool(best)))

    def _plan_coord(self, a: Agent):
        """Plan one vehicle around the current bookings (e.g. it just finished a job)."""
        t0 = self.t - 1
        path, reached, _ = self._st_search(a, t0, self.res, self.eres)
        self._book(a, path, reached, t0, self.res, self.eres)
        a.path = list(path)
        return True

    def _new_leg(self, a: Agent):
        if a.phase in ("to_pick", "packing"):
            a.phase = "to_pick"
            a.goal = a.rng.choice(self.lay.pick_cells)
            a.order_start = self.t
        else:
            a.phase = "to_pack"
            a.goal = a.rng.choice(self.lay.pack_cells)
        self._plan(a)

    def _reroute(self):
        """Humans re-choose selfishly (staggered); the planner re-plans the whole fleet."""
        k = self.p.reroute_every
        for a in self.agents:
            if a.kind == "human" and a.path and (self.t + a.id) % k == 0:
                self._plan(a)
        has_planner = any(a.kind == "coordinated" for a in self.agents)
        if has_planner and self.p.lookahead_every and self.t % self.p.lookahead_every == 0:
            self._lookahead()
        if has_planner and (self._need_replan or self.t % self.p.replan_every == 0):
            self._replan_fleet()

    def _leave(self, a: Agent):
        """Agent is leaving its cell: update the learned congestion map."""
        c, al = a.pos, self.p.alpha
        self.T[c] += al * ((self.t - a.enter_t) - self.T[c])
        self.E[c] += al * (a.caused - self.E[c])
        a.caused = 0
        a.enter_t = self.t

    # --- stepping ----------------------------------------------------------------
    def step(self):
        self.t += 1
        self._reroute()
        order = self.agents[:]
        self.rng.shuffle(order)

        # 1) dwell / arrival / (re)planning
        movers = []
        for a in order:
            if a.phase in ("picking", "packing"):
                a.dwell -= 1
                if a.dwell <= 0:
                    if a.phase == "packing":
                        self.completed += 1
                        self.cycle_times.append(self.t - a.order_start)
                    a.enter_t = self.t
                    a.caused = 0
                    self._new_leg(a)
                if a.kind != "coordinated" or a.phase in self.DWELLING:
                    continue                          # planned vehicles leave on schedule
            if not a.path:
                if a.pos == a.goal:
                    a.phase = "picking" if a.phase == "to_pick" else "packing"
                    a.dwell = self.p.pick_time if a.phase == "picking" else self.p.pack_time
                else:
                    self._plan(a)
                continue
            if a.path[0] == a.pos:                    # a planned wait
                a.path.pop(0)
                self.planned_waits += 1
                continue
            movers.append(a)

        # 2) resolve simultaneous moves: agents behind a moving agent follow it
        #    (queues move as a train) and closed loops of 4+ rotate
        want = {a.id: a.path[0] for a in movers}
        resolved: dict[int, bool] = {}
        claimed: set[int] = set()
        in_cycle: set[int] = set()

        def resolve(aid, stack):
            if aid in resolved:
                return resolved[aid]
            if aid in stack:                          # closed loop -> rotate
                cyc = stack[stack.index(aid):]
                if len(cyc) == 2:
                    return False                      # head-on: vehicles can't pass through each other
                for x in cyc:
                    resolved[x] = True
                    in_cycle.add(x)
                    claimed.add(want[x])
                return True
            tgt = want[aid]
            if tgt in claimed:
                resolved[aid] = False
                return False
            stack.append(aid)
            other = self.occ.get(tgt)
            if other is None:
                ok = True
            elif other in want:
                ok = resolve(other, stack) and other not in in_cycle
            else:
                ok = False                            # blocked by a dwelling agent
            stack.pop()
            if aid in resolved:                       # set while closing a loop
                return resolved[aid]
            if ok:
                claimed.add(tgt)
            resolved[aid] = ok
            return ok

        for a in movers:
            resolve(a.id, [])

        # 3) apply moves simultaneously
        moving = [a for a in movers if resolved[a.id]]
        for a in moving:
            self._leave(a)
            del self.occ[a.pos]
        for a in moving:
            a.pos = want[a.id]
            self.occ[a.pos] = a.id
            self.visits[a.pos] += 1
            self.move_steps += 1
            a.path.pop(0)
            a.blocked = 0

        # 4) blocked agents wait, replan around obstacles, or side-step if stuck
        for a in movers:
            if resolved[a.id]:
                continue
            a.blocked += 1
            self.blocked_steps += 1
            b = self.occ.get(want[a.id])
            if b is not None and self.agents[b].phase not in ("picking", "packing"):
                self.agents[b].caused += 1           # b's route choice is delaying a
            if a.kind == "coordinated":
                self._need_replan = True             # reality diverged from the plan
                if a.blocked < self.p.unstick:
                    continue
            if a.blocked >= self.p.unstick:
                free = [n for n in self.lay.neighbors[a.pos] if n not in self.occ]
                if free:
                    cell = a.rng.choice(free)
                    self._leave(a)
                    del self.occ[a.pos]
                    self.occ[cell] = a.id
                    a.pos = cell
                    self.move_steps += 1
                    self._plan(a)
                a.blocked = 0
            elif a.blocked >= self.p.patience:
                self._plan(a, avoid_occupied=True)

    def run(self, steps=1500, warmup=200):
        for _ in range(warmup):
            self.step()
        c0, b0, m0 = self.completed, self.blocked_steps, self.move_steps
        n0 = len(self.cycle_times)
        for _ in range(steps):
            self.step()
        done = self.completed - c0
        ct = self.cycle_times[n0:]
        blocked = self.blocked_steps - b0
        moves = self.move_steps - m0
        return {
            "throughput_per_1000": 1000 * done / steps,
            "mean_cycle_time": float(np.mean(ct)) if ct else float("nan"),
            "blocked_frac": blocked / max(1, blocked + moves),
        }
