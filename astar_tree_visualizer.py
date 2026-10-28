"""
A* SEARCH ALGORITHM - AUTOMATIC TREE VISUALIZER
AI and Soft Computing - Academic Project

How to run:
    python3 astar_tree_visualizer.py              -> live animation (press Start once)
    python3 astar_tree_visualizer.py --text       -> timed text-only trace (no graphics needed)
    python3 astar_tree_visualizer.py --save       -> export a Full-HD video astar_search.mp4
                                                     (needs ffmpeg:  brew install ffmpeg)
"""
import sys, math, time, heapq, textwrap
import numpy as np
import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")                      # exporting a video needs no window
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation, FFMpegWriter


# ============================ 1. THE PROBLEM ============================
def define_graph():
    """Return the search graph, the heuristic table and the fixed drawing positions."""
    graph = {                                  # adjacency dictionary: node -> [(child, edge cost)]
        "S": [("A", 2), ("B", 3)],
        "A": [("C", 2), ("D", 4)],
        "B": [("E", 2), ("F", 5)],
        "C": [("H", 3)],
        "D": [("G", 3)],
        "E": [("G", 5)],                       # second, more expensive route to G
        "F": [("I", 2)],
        "G": [], "H": [], "I": [],
    }
    heur = {"S": 7, "A": 5, "B": 4, "C": 3, "D": 3, "E": 3,   # h(n): admissible and consistent
            "F": 3, "G": 0, "H": 4, "I": 2}
    pos = {"S": (0, 4.5),
           "A": (-2.4, 3), "B": (2.4, 3),
           "C": (-3.6, 1.5), "D": (-1.2, 1.5), "E": (1.2, 1.5), "F": (3.6, 1.5),
           "H": (-3.6, 0), "G": (0, 0), "I": (3.6, 0)}
    return graph, heur, pos


GRAPH, HEUR, POS = define_graph()
START, GOAL = "S", "G"


def heuristic(n):
    """h(n): estimated remaining cost from n to the goal."""
    return HEUR[n]


# ============================ 2. THE A* ALGORITHM ============================
def astar_search(start, goal):
    """Create all A* data structures. The search itself runs inside astar_steps()."""
    st = {"open": [(heuristic(start), start)],  # priority queue (min-heap) of (f, node)
          "closed": set(),                      # nodes already expanded
          "order": [],                          # same nodes, in expansion order (for display)
          "g": {start: 0},                      # best known actual cost from start
          "parent": {start: None},              # parent pointers to rebuild the path
          "path": []}
    st["steps"] = astar_steps(st, goal)         # generator: one real A* operation per next()
    return st


def astar_steps(st, goal):
    """The genuine A* loop. Every 'yield' pauses the algorithm so the animation can show it."""
    open_list, closed, g, parent = st["open"], st["closed"], st["g"], st["parent"]
    yield ("init", START, None)
    while open_list:
        f, n = heapq.heappop(open_list)         # node with the LOWEST f(n)
        if n in closed:                         # outdated duplicate entry -> ignore it
            yield ("stale", n, f)
            continue
        closed.add(n)                           # move n from OPEN to CLOSED
        st["order"].append(n)
        if n == goal:                           # stop when the goal is SELECTED
            yield ("goal", n, None)
            st["path"] = reconstruct_path(parent, goal)
            for i in range(1, len(st["path"])):  # highlight the path edge by edge
                yield ("path", st["path"][i], st["path"][i - 1])
            yield ("done", goal, None)
            return
        yield ("select", n, None)
        for nb, cost in GRAPH[n]:               # examine every neighbour of n
            if nb in closed:
                yield ("skip", n, nb)
                continue
            new_g = g[n] + cost                 # tentative cost of reaching nb through n
            if new_g < g.get(nb, math.inf):     # first time seen OR a cheaper route
                old_g = g.get(nb)
                g[nb], parent[nb] = new_g, n
                heapq.heappush(open_list, (new_g + heuristic(nb), nb))   # f = g + h
                yield ("update" if old_g is not None else "discover", n, (nb, old_g))
            else:
                yield ("worse", n, (nb, new_g))
        yield ("expanded", n, None)
    yield ("fail", None, None)


def reconstruct_path(parent, node):
    """Follow parent pointers back to the start, then reverse."""
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def open_entries(st):
    """Split heap entries into valid ones and outdated duplicates (for display)."""
    valid, stale = [], []
    for f, n in sorted(st["open"]):
        ok = n not in st["closed"] and f == st["g"][n] + heuristic(n)
        (valid if ok else stale).append((f, n))
    return valid, stale


def describe(ev, st, step):
    """Turn one algorithm event into panel text: (title, value lines, explanation)."""
    kind, n, info = ev
    g = st["g"]
    f = lambda x: g[x] + heuristic(x)
    vals = lambda x: f"g({x}) = {g[x]}   h({x}) = {heuristic(x)}   f({x}) = {f(x)}"
    title, lines = f"STEP {step} — EXPANDING NODE {n}", [f"Current node: {n}"]
    if kind == "ready":
        return ("A* SEARCH — READY",
                [f"Start node: {START}     Goal node: {GOAL}", "", "f(n) = g(n) + h(n)",
                 "g(n) = actual cost from S to n", "h(n) = estimated cost from n to G",
                 "f(n) = estimated total cost through n"],
                "Press START once. The whole search then plays automatically, step by step.")
    if kind == "init":
        title, lines = "STEP 0 — INITIALISATION", [f"Start node: {n}", vals(n)]
        expl = (f"A* starts at {n}. g({n}) = 0 because nothing has been paid yet, h({n}) = "
                f"{heuristic(n)} estimates the rest, so f({n}) = {f(n)}. {n} is pushed into OPEN.")
    elif kind in ("select", "goal"):
        lines.append(vals(n))
        ties = [m for fv, m in open_entries(st)[0] if fv == f(n)]
        tie = f" (Tie with {', '.join(ties)}: heapq picks the alphabetically smaller name.)" if ties else ""
        expl = (f"A* removes {n} from OPEN because f({n}) = {f(n)} is the lowest f(n) in the "
                f"priority queue.{tie} {n} moves to CLOSED and its neighbours are examined next.")
        if kind == "goal":
            title = f"STEP {step} — GOAL {n} SELECTED"
            expl = (f"The goal {n} now has the lowest f(n) = {f(n)} in OPEN.{tie} With an admissible "
                    f"heuristic no other path can be cheaper, so A* stops. Optimal cost = {g[n]}.")
    elif kind in ("discover", "update"):
        nb, old = info
        title = f"STEP {step} — EXPANDING {n}: " + (f"NEW NODE {nb}" if kind == "discover" else "BETTER PATH")
        lines = [f"Current node: {n}   g({n}) = {g[n]}",
                 f"g({nb}) = g({n}) + cost({n},{nb}) = {g[n]} + {g[nb] - g[n]} = {g[nb]}",
                 f"h({nb}) = {heuristic(nb)}",
                 f"f({nb}) = g({nb}) + h({nb}) = {g[nb]} + {heuristic(nb)} = {f(nb)}"]
        if kind == "discover":
            expl = (f"{nb} is seen for the first time. A* stores g({nb}) = {g[nb]}, sets "
                    f"parent({nb}) = {n} and pushes {nb} into OPEN with priority f = {f(nb)}.")
        else:
            expl = (f"Cheaper route found! Through {n}, g({nb}) drops from {old} to {g[nb]}. A* updates "
                    f"g({nb}), sets parent({nb}) = {n} and pushes a new entry with f = {f(nb)}. "
                    f"The old entry stays in the heap and will be ignored.")
    elif kind == "worse":
        nb, new_g = info
        lines.append(f"Route via {n}: g = {new_g}   known g({nb}) = {g[nb]}")
        expl = f"Reaching {nb} through {n} costs {new_g}, not cheaper than {g[nb]}, so it is ignored."
    elif kind == "skip":
        expl = f"{info} is already in CLOSED (fully expanded), so it is skipped."
    elif kind == "stale":
        title, lines = f"STEP {step} — IGNORING OLD ENTRY", [f"Popped entry: {n} (f = {info})"]
        expl = f"{n} is already closed with a cheaper cost, so this outdated heap entry is discarded."
    elif kind == "expanded":
        title = f"STEP {step} — {n} FULLY EXPANDED"
        expl = (f"All neighbours of {n} have been examined. {n} stays in CLOSED (explored). "
                f"Next, A* picks the OPEN node with the lowest f(n).")
    elif kind == "path":
        u, path = info, st["path"]
        title = "RECONSTRUCTING THE OPTIMAL PATH"
        lines = [f"parent({n}) = {u}", "Path: " + " → ".join(path[:path.index(n) + 1])]
        expl = (f"A* follows parent pointers back from {GOAL} to {START} and reverses them. "
                f"Highlighting edge {u} → {n} (cost {g[n] - g[u]}).")
    elif kind == "done":
        path = st["path"]
        costs = " + ".join(str(g[b] - g[a]) for a, b in zip(path, path[1:]))
        title = "A* SEARCH COMPLETED —\nOPTIMAL PATH FOUND"
        lines = [f"Starting node : {START}", f"Goal node     : {GOAL}",
                 f"Optimal path  : {' → '.join(path)}", f"Total cost    : {costs} = {g[GOAL]}",
                 f"Expanded nodes: {len(st['order'])}  ({', '.join(st['order'])})",
                 "Status        : SUCCESS"]
        expl = ("Yellow = optimal path. Red nodes were explored but are not on the path; "
                "blue nodes were discovered but never needed; grey nodes were never reached.")
    else:
        title, lines, expl = "SEARCH FAILED", [], "OPEN became empty: the goal is unreachable."
    return title, lines, expl


def queue_text(st):
    """Live OPEN / CLOSED contents, read straight from the real data structures."""
    valid, stale = open_entries(st)
    open_txt = "   ".join(f"{m} (f={fv})" for fv, m in valid) or "empty"
    stale_txt = ("old heap entry, will be ignored: " + "  ".join(f"{m} (f={fv})" for fv, m in stale)
                 if stale else "")
    return open_txt, stale_txt, "  →  ".join(st["order"]) or "empty"


# ============================ 3. VISUALISATION ============================
BG, CARD, BORDER = "#0b1120", "#131c31", "#27344f"          # dark theme
TEXT, MUTED = "#e2e8f0", "#8b9bb4"
COL = {"start": "#22c55e", "unvisited": "#3b4a63", "open": "#3b82f6", "current": "#f97316",
       "closed": "#ef4444", "goal": "#a855f7", "path": "#facc15"}
ACCENT = {"ready": TEXT, "init": COL["start"], "select": COL["current"], "goal": COL["goal"],
          "discover": COL["open"], "update": COL["goal"], "worse": MUTED, "skip": MUTED,
          "stale": MUTED, "expanded": COL["closed"], "path": COL["path"],
          "done": COL["start"], "fail": COL["closed"]}
DURATION = {"init": 3.0, "select": 3.0, "goal": 3.5, "discover": 2.5, "update": 3.5,
            "worse": 2.0, "skip": 1.5, "stale": 2.0, "expanded": 1.8, "path": 1.3,
            "ready": math.inf, "done": math.inf, "fail": math.inf}     # seconds per event
CURRENT_KINDS = ("init", "select", "goal", "discover", "update", "worse", "skip")
R, TRAIL, TICK = 0.44, 16, 0.04                 # node radius, marker trail length, 25 fps
anim = {"speed": 1.0}                           # animation state
view = {}                                       # matplotlib objects we update


def setup_style():
    """Pick a clean font that exists on this computer (Inter / Avenir / Helvetica ...)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = [f for f in ("Inter", "Avenir Next", "Helvetica Neue", "Segoe UI", "Arial") if f in names]
    mono = [f for f in ("JetBrains Mono", "Menlo", "SF Mono", "Consolas") if f in names]
    plt.rcParams.update({"font.family": sans + ["DejaVu Sans"], "text.color": TEXT,
                         "font.monospace": mono + ["DejaVu Sans Mono"]})


def node_color(n):
    """Colour of node n, derived directly from the real A* data structures."""
    st, (kind, cur, _) = anim["st"], anim["ev"]
    if any(n in e for e in anim["path_edges"]):
        return COL["path"]
    if n == GOAL and n in st["g"]:
        return COL["goal"]
    if n == cur and kind in CURRENT_KINDS:
        return COL["current"]
    if n == START:
        return COL["start"]
    if n in st["closed"]:
        return COL["closed"]
    return COL["open"] if n in st["g"] else COL["unvisited"]   # discovered = waiting in OPEN


def styled_button(fig, rect, label, color, hover):
    bax = fig.add_axes(rect)
    btn = Button(bax, label, color=color, hovercolor=hover)
    btn.label.set(color="white", fontweight="bold", fontsize=12)
    for spine in bax.spines.values():
        spine.set_visible(False)
    return btn


def draw_tree(dpi=90):
    """Draw the fixed tree, the explanation panel and the controls (only once)."""
    setup_style()
    fig = view["fig"] = plt.figure(figsize=(16, 9), dpi=dpi, facecolor=BG)
    fig.text(0.025, 0.935, "A* SEARCH", fontsize=28, fontweight="bold", color="white")
    fig.text(0.025, 0.905, "Informed search   ·   f(n) = g(n) + h(n)   ·   AI & Soft Computing",
             fontsize=12, color=MUTED)
    view["status"] = fig.text(0.615, 0.93, "", ha="right", fontsize=13, fontweight="bold",
                              bbox=dict(boxstyle="round,pad=0.55", fc=CARD, ec=BORDER, lw=1.5))
    ax = fig.add_axes([0.02, 0.15, 0.60, 0.74], facecolor=BG)
    ax.set_xlim(-4.4, 5.1); ax.set_ylim(-0.75, 5.0); ax.set_aspect("equal"); ax.axis("off")
    view.update(arrows={}, glow={}, nodes={}, halos={}, labels={}, gf={})
    for u in GRAPH:                             # ---- edges as arrows, with cost labels
        for v, cost in GRAPH[u]:
            (x1, y1), (x2, y2) = POS[u], POS[v]
            d = math.hypot(x2 - x1, y2 - y1)
            ux, uy = (x2 - x1) / d, (y2 - y1) / d
            a, b = (x1 + ux * R, y1 + uy * R), (x2 - ux * (R + 0.03), y2 - uy * (R + 0.03))
            view["glow"][(u, v)], = ax.plot([a[0], b[0]], [a[1], b[1]], color=COL["path"], lw=18,
                                            alpha=0.18, solid_capstyle="round", zorder=1)
            view["arrows"][(u, v)] = ax.add_patch(FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=22, shrinkA=0, shrinkB=0, zorder=2))
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(cost), ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.32", fc=CARD, ec="#475569", lw=1.2))
    for n, (x, y) in POS.items():               # ---- glowing nodes
        view["halos"][n] = [ax.add_patch(Circle((x, y), r, ec="none", alpha=al, zorder=2))
                            for r, al in ((0.74, 0.10), (0.6, 0.22))]
        view["nodes"][n] = ax.add_patch(Circle((x, y), R, lw=2, zorder=4))
        view["labels"][n] = [
            ax.text(x, y + 0.09, n, ha="center", va="center", fontsize=18, fontweight="bold", zorder=5),
            ax.text(x, y - 0.2, f"h = {heuristic(n)}", ha="center", va="center", fontsize=9.5, zorder=5)]
        view["gf"][n] = ax.text(x + 0.56, y, "", ha="left", va="center", fontsize=10.5, zorder=5,
                                fontweight="bold", color=TEXT,
                                bbox=dict(boxstyle="round,pad=0.35", fc=CARD, ec=BORDER, lw=1.5))
    view["ring"] = ax.add_patch(Circle((0, 0), 0.56, fill=False, lw=3.5, zorder=3))     # current node
    view["ripple"] = ax.add_patch(Circle((0, 0), 0.5, fill=False, lw=2.5, zorder=3))
    view["trail"] = ax.scatter([], [], s=[], zorder=6, edgecolors="none")               # moving marker
    view["mglow"], = ax.plot([], [], "o", ms=32, alpha=0.35, mec="none", zorder=6)
    view["mcore"], = ax.plot([], [], "o", ms=13, mec="white", mew=2.2, zorder=7)
    ax.text(-4.35, 4.95, "Inside circle: node, h(n)\nBeside circle: g(n), f(n)\nOn arrow: step cost",
            fontsize=10, va="top", color=MUTED, linespacing=1.6)
    labels = ["Start", "Unvisited", "Discovered (OPEN)", "Current (expanding)",
              "Explored (CLOSED)", "Goal", "Optimal path"]
    handles = [Line2D([0], [0], marker="o", ls="", ms=14, mfc=c, mec="#e2e8f0", mew=1.2, label=l)
               for c, l in zip(COL.values(), labels)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False,
              fontsize=11, labelcolor=TEXT, columnspacing=2.0, handletextpad=0.5)

    pax = fig.add_axes([0.64, 0.15, 0.345, 0.79]); pax.axis("off")   # ---- explanation panel
    pax.set(xlim=(0, 1), ylim=(0, 1)); pax.set_autoscale_on(False)
    pax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.025",
                                 fc=CARD, ec=BORDER, lw=1.5, transform=pax.transAxes))
    T = lambda y, size, color, **kw: pax.text(0.05, y, "", fontsize=size, color=color, va="top", **kw)
    view["ptitle"] = T(0.955, 14.5, TEXT, fontweight="bold", linespacing=1.3)
    view["divider"], = pax.plot([0.05, 0.95], [0.862, 0.862], lw=2.5, solid_capstyle="round")
    view["calc_hdr"] = T(0.835, 9.5, MUTED, fontweight="bold")
    view["calc"] = T(0.80, 11, TEXT, family="monospace", linespacing=1.55)
    T(0.585, 9.5, MUTED, fontweight="bold").set_text("OPEN LIST  ·  PRIORITY QUEUE (lowest f first)")
    view["open"] = T(0.552, 11, "#93c5fd", family="monospace", linespacing=1.4)
    view["stale"] = T(0.505, 9.5, MUTED, style="italic")
    T(0.465, 9.5, MUTED, fontweight="bold").set_text("CLOSED SET  ·  already expanded")
    view["closed"] = T(0.432, 11, "#fca5a5", family="monospace")
    pax.plot([0.05, 0.95], [0.375, 0.375], lw=1, color=BORDER)
    T(0.35, 9.5, MUTED, fontweight="bold").set_text("WHY THIS STEP?")
    view["pexpl"] = T(0.315, 11.5, TEXT, linespacing=1.5)

    view["bstart"] = styled_button(fig, [0.64, 0.035, 0.105, 0.06], "Start", "#16a34a", "#22c55e")
    view["bpause"] = styled_button(fig, [0.76, 0.035, 0.105, 0.06], "Pause", "#334155", "#475569")
    view["breset"] = styled_button(fig, [0.88, 0.035, 0.105, 0.06], "Reset", "#334155", "#475569")
    sax = fig.add_axes([0.09, 0.045, 0.30, 0.025], facecolor=BG)
    view["speed"] = Slider(sax, "Speed ", 0.25, 3.0, valinit=1.0, valstep=0.25, valfmt="%.2fx",
                           color=COL["current"], track_color=BORDER, initcolor="none",
                           handle_style={"facecolor": "white", "edgecolor": COL["current"], "size": 14})
    view["speed"].label.set(color=TEXT, fontsize=11)
    view["speed"].valtext.set(color=TEXT, fontsize=11)
    view["widgets"] = [view[k].ax for k in ("bstart", "bpause", "breset", "speed")]
    view["bstart"].on_clicked(start_animation)
    view["bpause"].on_clicked(toggle_pause)
    view["breset"].on_clicked(reset_visualization)
    view["speed"].on_changed(lambda val: anim.update(speed=val))


def refresh_view():
    """Update colours, cost labels, arrows and panel text for the current algorithm event."""
    st, g, kind = anim["st"], anim["st"]["g"], anim["ev"][0]
    for n in POS:
        c = node_color(n)
        view["nodes"][n].set(facecolor=c, edgecolor="#64748b" if c == COL["unvisited"] else "#f8fafc")
        for halo in view["halos"][n]:           # soft glow around every reached node
            halo.set(facecolor=c, visible=c != COL["unvisited"])
        for t in view["labels"][n]:             # dark text on yellow, white elsewhere
            t.set_color("#1c1917" if c == COL["path"] else "white")
        view["gf"][n].set_text(f"g = {g[n]}   f = {g[n] + heuristic(n)}" if n in g else "")
        view["gf"][n].get_bbox_patch().set_edgecolor(c)
    for (u, v), arrow in view["arrows"].items():
        on_path = (u, v) in anim["path_edges"]
        view["glow"][(u, v)].set_visible(on_path)
        if on_path:
            arrow.set(color=COL["path"], lw=4.5)         # optimal path
        elif st["parent"].get(v) == u:
            arrow.set(color="#cbd5e1", lw=2.6)           # current best way to reach v
        else:
            arrow.set(color="#334155", lw=1.8)           # not (or no longer) used
    title, lines, expl = describe(anim["ev"], st, anim["step"])
    accent = ACCENT[kind]
    view["ptitle"].set(text="\n".join(textwrap.fill(p, 36) for p in title.split("\n")), color=accent)
    view["divider"].set_color(accent)
    view["calc_hdr"].set_text({"done": "RESULT", "ready": "THE FORMULA"}.get(kind, "CALCULATION"))
    view["calc"].set_text("\n".join(textwrap.fill(l, 50, subsequent_indent="      ") for l in lines))
    open_txt, stale_txt, closed_txt = queue_text(st)
    view["open"].set_text(textwrap.fill(open_txt, 46))
    view["stale"].set_text(stale_txt)
    view["closed"].set_text(textwrap.fill(closed_txt, 46))
    view["pexpl"].set_text(textwrap.fill(expl, 48))
    status = {"ready": "READY  ·  PRESS START", "done": f"OPTIMAL PATH FOUND  ·  COST {g.get(GOAL)}",
              "path": "RECONSTRUCTING PATH"}.get(kind, f"SEARCHING  ·  STEP {anim['step']}")
    view["status"].set(text=status, color=accent if kind != "ready" else TEXT)
    view["status"].get_bbox_patch().set_edgecolor(accent)
    view["fig"].canvas.draw_idle()


def marker_route(ev, st):
    """Nodes the moving marker travels through for this event (real parent pointers)."""
    kind, n, info = ev
    if kind in ("init", "select", "goal"):
        return reconstruct_path(st["parent"], n)  # the path A* is currently extending
    if kind in ("discover", "update", "worse"):
        return [n, info[0]]
    return {"skip": [n, info], "path": [info, n]}.get(kind, [])


def advance():
    """Ask the algorithm for its next real operation and show it."""
    ev = next(anim["st"]["steps"])
    if ev[0] in ("select", "goal", "stale"):
        anim["step"] += 1
    if ev[0] == "path":
        anim["path_edges"].add((ev[2], ev[1]))
    anim.update(ev=ev, t=0.0, done=ev[0] in ("done", "fail"), trail=[],
                route=marker_route(ev, anim["st"]))
    refresh_view()


def update_animation(frame, dt=None):
    """Called every frame by FuncAnimation: advance time, move marker, pulse the rings."""
    now = time.perf_counter()
    if dt is None:                              # live mode: use the real elapsed time
        dt = min(now - anim.get("last", now), 0.1)
    anim["last"] = now
    if anim["running"] and not anim["paused"] and not anim["done"]:
        anim["t"] += dt * anim["speed"]
        if anim["t"] >= DURATION[anim["ev"][0]]:
            advance()                           # time for the next A* operation
    kind, n, _ = anim["ev"]
    route, t = anim["route"], anim["t"]
    frac = min(1.0, t / (0.6 * DURATION[kind])) if route else 1.0
    if route and frac < 1.0:                    # glowing marker glides along the arrows
        frac = frac * frac * (3 - 2 * frac)     # ease-in / ease-out
        seg = frac * (len(route) - 1)
        i = min(int(seg), max(len(route) - 2, 0))
        (x1, y1), (x2, y2) = POS[route[i]], POS[route[min(i + 1, len(route) - 1)]]
        x, y = x1 + (x2 - x1) * (seg - i), y1 + (y2 - y1) * (seg - i)
        color = {"discover": COL["open"], "update": COL["goal"], "path": COL["path"],
                 "worse": MUTED, "skip": MUTED}.get(kind, COL["current"])
        anim["trail"] = (anim["trail"] + [(x, y)])[-TRAIL:]
        k = len(anim["trail"])
        view["trail"].set_offsets(anim["trail"])
        view["trail"].set_sizes([30 + 150 * j / TRAIL for j in range(k)])
        view["trail"].set_facecolors([to_rgba(color, 0.05 + 0.4 * j / TRAIL) for j in range(k)])
        for m in (view["mglow"], view["mcore"]):
            m.set_data([x], [y])
            m.set_markerfacecolor(color)
    else:
        for m in (view["mglow"], view["mcore"]):
            m.set_data([], [])
        view["trail"].set_offsets(np.empty((0, 2)))
    active = kind in CURRENT_KINDS
    view["ring"].set_visible(active)
    view["ripple"].set_visible(active)
    if active:                                  # pulsing ring + expanding ripple on current node
        c = COL["goal"] if n == GOAL else COL["current"]
        phase = (t % 1.2) / 1.2
        view["ring"].set(center=POS[n], radius=0.57 + 0.03 * math.sin(t * 6), edgecolor=c)
        view["ripple"].set(center=POS[n], radius=0.5 + 0.45 * phase, edgecolor=c,
                           alpha=0.8 * (1 - phase))


def start_animation(event=None):
    if anim["done"]:
        reset_visualization()
    if not anim["running"]:
        anim["running"] = True
        advance()                               # first real event: initialisation


def toggle_pause(event=None):
    if anim["running"] and not anim["done"]:
        anim["paused"] = not anim["paused"]
        view["bpause"].label.set_text("Resume" if anim["paused"] else "Pause")


def reset_visualization(event=None):
    anim.update(st=astar_search(START, GOAL), ev=("ready", None, None), t=0.0, step=0, trail=[],
                running=False, paused=False, done=False, path_edges=set(), route=[])
    view["bpause"].label.set_text("Pause")
    refresh_view()


# ============================ 4. TEXT MODE AND VIDEO EXPORT ============================
def text_mode(delay=1.5):
    """Fallback without graphics: prints every real A* step automatically with a pause."""
    st, step = astar_search(START, GOAL), 0
    for ev in st["steps"]:
        step += ev[0] in ("select", "goal", "stale")
        title, lines, expl = describe(ev, st, step)
        open_txt, stale_txt, closed_txt = queue_text(st)
        print("\n" + "=" * 60 + "\n" + title + "\n" + "\n".join(lines))
        print(f"OPEN  : {open_txt}  {stale_txt}\nCLOSED: {closed_txt}\n-> {expl}")
        time.sleep(delay)


def save_video(fps=30):
    """Play the whole search off-screen and save it as a Full-HD (1920x1080) MP4."""
    if not FFMpegWriter.isAvailable():
        print("ffmpeg is needed for video export. Install it with:  brew install ffmpeg")
        return
    for wax in view["widgets"]:                 # buttons are not needed in a video
        wax.set_visible(False)

    def frames():                               # 2 s intro, the search, then 5 s on the result
        for _ in range(2 * fps):
            yield 0
        start_animation()
        hold = 0
        while hold < 5 * fps:
            hold += anim["done"]
            yield 0
    ani = FuncAnimation(view["fig"], lambda fr: update_animation(fr, 1 / fps), frames=frames,
                        save_count=100000, cache_frame_data=False)
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "17", "-preset", "slow"])
    ani.save("astar_search.mp4", writer=writer, dpi=120, savefig_kwargs={"facecolor": BG})
    print("Saved astar_search.mp4 (1920x1080)")


if __name__ == "__main__":
    if "--text" in sys.argv:
        text_mode()
        sys.exit()
    draw_tree()
    reset_visualization()
    if "--save" in sys.argv:
        save_video()
    else:
        view["ani"] = FuncAnimation(view["fig"], update_animation, interval=int(TICK * 1000),
                                    cache_frame_data=False)
        plt.show()
