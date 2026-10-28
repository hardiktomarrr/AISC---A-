"""
Streamlit Web App: A* Search Tree Visualizer (Fast, Premium Design & Zero-Lag)
AI and Soft Computing Academic Project
"""
import streamlit as st
import streamlit.components.v1 as components
import time
import copy
import heapq
import math

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="A* Search Tree Visualizer",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling CSS
st.markdown("""
<style>
    /* Global Theme Overrides */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Metrics Header Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 12px 18px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
    }
    
    /* Custom Info Cards */
    .status-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }
    
    .status-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    
    .queue-chip {
        display: inline-block;
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid #3b82f6;
        padding: 4px 10px;
        border-radius: 20px;
        font-family: monospace;
        font-size: 0.88rem;
        font-weight: 600;
        margin: 3px;
    }
    
    .stale-chip {
        display: inline-block;
        background: #311313;
        color: #f87171;
        border: 1px dashed #ef4444;
        padding: 3px 8px;
        border-radius: 20px;
        font-family: monospace;
        font-size: 0.82rem;
        margin: 2px;
    }
    
    .closed-chip {
        display: inline-block;
        background: #1e1e2f;
        color: #cbd5e1;
        border: 1px solid #4b5563;
        padding: 3px 9px;
        border-radius: 6px;
        font-family: monospace;
        font-weight: bold;
        font-size: 0.85rem;
        margin: 2px;
    }
    
    .calc-row {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        background: #090d16;
        padding: 8px 12px;
        border-radius: 6px;
        border-left: 3px solid #38bdf8;
        color: #e2e8f0;
        font-size: 0.88rem;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ============================ 1. THE PROBLEM DEFINITION ============================
GRAPH = {
    "S": [("A", 2), ("B", 3)],
    "A": [("C", 2), ("D", 4)],
    "B": [("E", 2), ("F", 5)],
    "C": [("H", 3)],
    "D": [("G", 3)],
    "E": [("G", 5)],
    "F": [("I", 2)],
    "G": [], "H": [], "I": []
}

HEUR = {
    "S": 7, "A": 5, "B": 4, "C": 3, "D": 3, "E": 3,
    "F": 3, "G": 0, "H": 4, "I": 2
}

# Enlarged SVG coordinates (canvas: 960x560)
NODE_COORDS = {
    "S": (480, 68),
    "A": (240, 195),
    "B": (720, 195),
    "C": (120, 335),
    "D": (360, 335),
    "E": (600, 335),
    "F": (840, 335),
    "H": (120, 485),
    "G": (480, 485),
    "I": (840, 485)
}

START, GOAL = "S", "G"

def reconstruct_path(parent, node):
    path = []
    while node is not None:
        path.append(node)
        node = parent.get(node)
    return path[::-1]

def open_entries(open_list, closed_set, g_dict):
    valid, stale = [], []
    for f, n in sorted(open_list):
        ok = (n not in closed_set) and (f == g_dict[n] + HEUR[n])
        (valid if ok else stale).append((f, n))
    return valid, stale

# ============================ 2. PRECOMPUTE ALGORITHM SNAPSHOTS ============================
@st.cache_data
def compute_all_steps():
    """Run A* and capture the full state at each step for instant scrub/playback."""
    st_dict = {
        "open": [(HEUR[START], START)],
        "closed": set(),
        "order": [],
        "g": {START: 0},
        "parent": {START: None},
        "path": []
    }
    
    snapshots = []
    step_num = 0
    path_edges = []
    
    # Initial ready state
    snapshots.append({
        "step_num": 0,
        "event_kind": "ready",
        "current_node": START,
        "active_edge": None,
        "path_edges": [],
        "open_valid": [(HEUR[START], START)],
        "open_stale": [],
        "closed": set(),
        "order": [],
        "g": {START: 0},
        "parent": {START: None},
        "title": "A* SEARCH — READY",
        "metrics": {"node": START, "g": 0, "h": HEUR[START], "f": HEUR[START]},
        "expl": "Press **Next ⏭** or **▶ Auto Play** to begin. The start node S will be expanded first.",
        "calc_lines": [f"Start node: {START} | Goal node: {GOAL}", f"f(S) = g(S) + h(S) = 0 + {HEUR[START]} = {HEUR[START]}"]
    })

    open_list = st_dict["open"]
    closed = st_dict["closed"]
    g = st_dict["g"]
    parent = st_dict["parent"]

    while open_list:
        f, n = heapq.heappop(open_list)
        if n in closed:
            valid, stale = open_entries(open_list, closed, g)
            snapshots.append({
                "step_num": step_num,
                "event_kind": "stale",
                "current_node": n,
                "active_edge": None,
                "path_edges": list(path_edges),
                "open_valid": valid,
                "open_stale": stale,
                "closed": set(closed),
                "order": list(st_dict["order"]),
                "g": dict(g),
                "parent": dict(parent),
                "title": f"IGNORING OUTDATED HEAP ENTRY: {n} (f={f})",
                "metrics": {"node": n, "g": g[n], "h": HEUR[n], "f": f},
                "expl": f"Node {n} was already closed with a cheaper route. This duplicate entry with f={f} is discarded via lazy deletion.",
                "calc_lines": [f"Popped stale entry: ({f}, '{n}')", f"Known best cost g({n}) = {g[n]}"]
            })
            continue

        step_num += 1
        closed.add(n)
        st_dict["order"].append(n)
        valid, stale = open_entries(open_list, closed, g)

        if n == GOAL:
            final_path = reconstruct_path(parent, GOAL)
            snapshots.append({
                "step_num": step_num,
                "event_kind": "goal",
                "current_node": n,
                "active_edge": None,
                "path_edges": list(path_edges),
                "open_valid": valid,
                "open_stale": stale,
                "closed": set(closed),
                "order": list(st_dict["order"]),
                "g": dict(g),
                "parent": dict(parent),
                "title": f"STEP {step_num} — GOAL {n} SELECTED FROM OPEN LIST",
                "metrics": {"node": n, "g": g[n], "h": 0, "f": g[n]},
                "expl": f"The goal {n} now has the lowest f = {g[n]} in the priority queue. Since h(n) is admissible, no remaining path can be cheaper. A* terminates!",
                "calc_lines": [f"Selected: {n}", f"g({n}) = {g[n]}, h({n}) = 0, f({n}) = {g[n]}"]
            })
            
            # Reconstruct path step by step
            for i in range(1, len(final_path)):
                u, v = final_path[i-1], final_path[i]
                path_edges.append((u, v))
                snapshots.append({
                    "step_num": step_num,
                    "event_kind": "path",
                    "current_node": v,
                    "active_edge": (u, v),
                    "path_edges": list(path_edges),
                    "open_valid": valid,
                    "open_stale": stale,
                    "closed": set(closed),
                    "order": list(st_dict["order"]),
                    "g": dict(g),
                    "parent": dict(parent),
                    "title": f"RECONSTRUCTING OPTIMAL PATH: {u} → {v}",
                    "metrics": {"node": v, "g": g[v], "h": HEUR[v], "f": g[v] + HEUR[v]},
                    "expl": f"Tracing parent pointers backwards: parent({v}) = {u}. Highlighting edge {u} → {v}.",
                    "calc_lines": [f"Backtracking: {' → '.join(final_path[:i+1])}"]
                })

            snapshots.append({
                "step_num": step_num,
                "event_kind": "done",
                "current_node": GOAL,
                "active_edge": None,
                "path_edges": list(path_edges),
                "open_valid": valid,
                "open_stale": stale,
                "closed": set(closed),
                "order": list(st_dict["order"]),
                "g": dict(g),
                "parent": dict(parent),
                "title": "A* SEARCH COMPLETE — OPTIMAL PATH FOUND!",
                "metrics": {"node": GOAL, "g": g[GOAL], "h": 0, "f": g[GOAL]},
                "expl": f"Optimal Path: {' → '.join(final_path)} with total cost {g[GOAL]}. Explored {len(closed)} nodes in total.",
                "calc_lines": [
                    f"Path: {' → '.join(final_path)}",
                    f"Total Cost: {g[GOAL]}",
                    f"Expanded nodes: {', '.join(st_dict['order'])}"
                ]
            })
            break

        # Select event
        snapshots.append({
            "step_num": step_num,
            "event_kind": "select",
            "current_node": n,
            "active_edge": None,
            "path_edges": list(path_edges),
            "open_valid": valid,
            "open_stale": stale,
            "closed": set(closed),
            "order": list(st_dict["order"]),
            "g": dict(g),
            "parent": dict(parent),
            "title": f"STEP {step_num} — EXPANDING NODE {n}",
            "metrics": {"node": n, "g": g[n], "h": HEUR[n], "f": g[n] + HEUR[n]},
            "expl": f"Node {n} was popped from OPEN because it has the lowest f = {g[n] + HEUR[n]}. It moves into CLOSED, and its outgoing neighbours will now be evaluated.",
            "calc_lines": [f"Selected: {n}", f"g({n}) = {g[n]}, h({n}) = {HEUR[n]} → f({n}) = {g[n] + HEUR[n]}"]
        })

        # Neighbours
        for nb, cost in GRAPH[n]:
            if nb in closed:
                snapshots.append({
                    "step_num": step_num,
                    "event_kind": "skip",
                    "current_node": n,
                    "active_edge": (n, nb),
                    "path_edges": list(path_edges),
                    "open_valid": valid,
                    "open_stale": stale,
                    "closed": set(closed),
                    "order": list(st_dict["order"]),
                    "g": dict(g),
                    "parent": dict(parent),
                    "title": f"STEP {step_num} — NEIGHBOUR {nb} ALREADY CLOSED",
                    "metrics": {"node": nb, "g": g[nb], "h": HEUR[nb], "f": g[nb] + HEUR[nb]},
                    "expl": f"Node {nb} is already in CLOSED (fully explored). Consistent heuristic guarantees its cost cannot be improved, so it is skipped.",
                    "calc_lines": [f"Examining edge {n} → {nb} (cost {cost})", f"{nb} ∈ CLOSED → SKIPPED"]
                })
                continue

            new_g = g[n] + cost
            if new_g < g.get(nb, math.inf):
                old_g = g.get(nb)
                g[nb] = new_g
                parent[nb] = n
                heapq.heappush(open_list, (new_g + HEUR[nb], nb))
                valid, stale = open_entries(open_list, closed, g)

                if old_g is not None:
                    # Better path update
                    snapshots.append({
                        "step_num": step_num,
                        "event_kind": "update",
                        "current_node": n,
                        "active_edge": (n, nb),
                        "path_edges": list(path_edges),
                        "open_valid": valid,
                        "open_stale": stale,
                        "closed": set(closed),
                        "order": list(st_dict["order"]),
                        "g": dict(g),
                        "parent": dict(parent),
                        "title": f"STEP {step_num} — BETTER PATH TO {nb} FOUND!",
                        "metrics": {"node": nb, "g": new_g, "h": HEUR[nb], "f": new_g + HEUR[nb]},
                        "expl": f"Cheaper route found! Through {n}, g({nb}) dropped from {old_g} to {new_g}. parent({nb}) is updated to {n}, and new entry ({new_g + HEUR[nb]}, '{nb}') is pushed to OPEN.",
                        "calc_lines": [
                            f"New g({nb}) = g({n}) + cost({n},{nb}) = {g[n]} + {cost} = {new_g} (was {old_g})",
                            f"h({nb}) = {HEUR[nb]}",
                            f"New f({nb}) = {new_g} + {HEUR[nb]} = {new_g + HEUR[nb]}"
                        ]
                    })
                else:
                    # Discovered new node
                    snapshots.append({
                        "step_num": step_num,
                        "event_kind": "discover",
                        "current_node": n,
                        "active_edge": (n, nb),
                        "path_edges": list(path_edges),
                        "open_valid": valid,
                        "open_stale": stale,
                        "closed": set(closed),
                        "order": list(st_dict["order"]),
                        "g": dict(g),
                        "parent": dict(parent),
                        "title": f"STEP {step_num} — DISCOVERED NEW NODE {nb}",
                        "metrics": {"node": nb, "g": new_g, "h": HEUR[nb], "f": new_g + HEUR[nb]},
                        "expl": f"Node {nb} is seen for the first time. Set g({nb}) = {new_g}, parent({nb}) = {n}, and push into OPEN with priority f = {new_g + HEUR[nb]}.",
                        "calc_lines": [
                            f"g({nb}) = g({n}) + cost({n},{nb}) = {g[n]} + {cost} = {new_g}",
                            f"h({nb}) = {HEUR[nb]}",
                            f"f({nb}) = g({nb}) + h({nb}) = {new_g} + {HEUR[nb]} = {new_g + HEUR[nb]}"
                        ]
                    })
            else:
                snapshots.append({
                    "step_num": step_num,
                    "event_kind": "worse",
                    "current_node": n,
                    "active_edge": (n, nb),
                    "path_edges": list(path_edges),
                    "open_valid": valid,
                    "open_stale": stale,
                    "closed": set(closed),
                    "order": list(st_dict["order"]),
                    "g": dict(g),
                    "parent": dict(parent),
                    "title": f"STEP {step_num} — WORSE ROUTE TO {nb} IGNORED",
                    "metrics": {"node": nb, "g": g[nb], "h": HEUR[nb], "f": g[nb] + HEUR[nb]},
                    "expl": f"Reaching {nb} via {n} costs {new_g}, which is not cheaper than existing cost {g[nb]}. It is ignored.",
                    "calc_lines": [f"Candidate route: g = {new_g} ≥ current g({nb}) = {g[nb]}"]
                })

    return snapshots

SNAPSHOTS = compute_all_steps()
TOTAL_STEPS = len(SNAPSHOTS) - 1

# ============================ 3. ENLARGED SVG TREE RENDERING ============================
def render_svg_tree(snap):
    """Generate crisp, hardware-accelerated SVG with modern styling and enlarged canvas."""
    curr_n = snap["current_node"]
    closed_set = snap["closed"]
    open_nodes = {n for f, n in snap["open_valid"]}
    path_edges = set(snap["path_edges"])
    active_edge = snap["active_edge"]
    g_dict = snap["g"]
    is_done = snap["event_kind"] == "done"
    
    path_nodes = set()
    if path_edges:
        for u, v in path_edges:
            path_nodes.add(u)
            path_nodes.add(v)
    elif is_done:
        path_nodes = {"S", "A", "D", "G"}

    svg = []
    
    # SVG Definition & Styles (960x560 canvas)
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 560" style="width:100%; height:100%; background: #0b0f19; border-radius: 14px; border: 1px solid #1e293b; box-shadow: 0 10px 35px rgba(0,0,0,0.6);">')
    svg.append('<defs>')
    svg.append('<marker id="arrow" viewBox="0 0 10 10" refX="32" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#475569"/></marker>')
    svg.append('<marker id="arrow-gold" viewBox="0 0 10 10" refX="32" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#fbbf24"/></marker>')
    svg.append('<marker id="arrow-active" viewBox="0 0 10 10" refX="32" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#f97316"/></marker>')
    svg.append('<filter id="glow-gold" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="8" result="blur" /><feComposite in="SourceGraphic" in2="blur" operator="over" /></filter>')
    svg.append('<filter id="glow-orange" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="9" result="blur" /><feComposite in="SourceGraphic" in2="blur" operator="over" /></filter>')
    svg.append('</defs>')

    # Draw Edges
    for u, neighbours in GRAPH.items():
        x1, y1 = NODE_COORDS[u]
        for v, cost in neighbours:
            x2, y2 = NODE_COORDS[v]
            
            is_path = (u, v) in path_edges or (is_done and (u, v) in [("S","A"), ("A","D"), ("D","G")])
            is_active = (active_edge == (u, v))
            
            if is_path:
                stroke_color = "#fbbf24"
                stroke_width = "4.5"
                marker = "url(#arrow-gold)"
                filter_attr = 'filter="url(#glow-gold)"'
            elif is_active:
                stroke_color = "#f97316"
                stroke_width = "4"
                marker = "url(#arrow-active)"
                filter_attr = 'filter="url(#glow-orange)"'
            else:
                stroke_color = "#334155"
                stroke_width = "2.4"
                marker = "url(#arrow)"
                filter_attr = ""

            svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke_color}" stroke-width="{stroke_width}" marker-end="{marker}" {filter_attr}/>')
            
            # Edge Cost Label
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ox, oy = (my - y1) * 0.08, (x1 - mx) * 0.08
            svg.append(f'<rect x="{mx + ox - 14}" y="{my + oy - 12}" width="28" height="24" rx="6" fill="#0f172a" stroke="{stroke_color}" stroke-width="1.2"/>')
            svg.append(f'<text x="{mx + ox}" y="{my + oy + 4}" fill="#f1f5f9" font-size="13" font-weight="bold" font-family="system-ui, sans-serif" text-anchor="middle">{cost}</text>')

    # Draw Nodes
    for node, (x, y) in NODE_COORDS.items():
        h_val = HEUR[node]
        known_g = g_dict.get(node)
        
        # Determine Node Theme Color
        if is_done and node in path_nodes:
            bg_color = "#d97706"
            border_color = "#fbbf24"
            text_color = "#000"
            glow = 'filter="url(#glow-gold)"'
        elif node == curr_n and snap["event_kind"] not in ("ready", "done"):
            bg_color = "#ea580c"
            border_color = "#fed7aa"
            text_color = "#fff"
            glow = 'filter="url(#glow-orange)"'
        elif node in path_nodes:
            bg_color = "#d97706"
            border_color = "#fbbf24"
            text_color = "#000"
            glow = 'filter="url(#glow-gold)"'
        elif node in closed_set:
            bg_color = "#991b1b"
            border_color = "#f87171"
            text_color = "#fff"
            glow = ""
        elif node in open_nodes:
            bg_color = "#1d4ed8"
            border_color = "#60a5fa"
            text_color = "#fff"
            glow = ""
        elif node == GOAL and known_g is not None:
            bg_color = "#7c3aed"
            border_color = "#c084fc"
            text_color = "#fff"
            glow = ""
        else:
            bg_color = "#1e293b"
            border_color = "#475569"
            text_color = "#94a3b8"
            glow = ""

        # Outer Pulsing Ring for Active Node
        if node == curr_n and snap["event_kind"] not in ("ready", "done"):
            svg.append(f'<circle cx="{x}" cy="{y}" r="38" fill="none" stroke="#f97316" stroke-width="2.5" stroke-dasharray="5,5" opacity="0.9"/>')

        # Node Circle (enlarged r=28)
        svg.append(f'<circle cx="{x}" cy="{y}" r="28" fill="{bg_color}" stroke="{border_color}" stroke-width="3" {glow}/>')
        
        # Node Name & Heuristic inside circle
        svg.append(f'<text x="{x}" y="{y - 2}" fill="{text_color}" font-size="17" font-weight="900" font-family="system-ui, sans-serif" text-anchor="middle">{node}</text>')
        svg.append(f'<text x="{x}" y="{y + 15}" fill="{text_color}" font-size="11" font-weight="600" font-family="system-ui, sans-serif" text-anchor="middle" opacity="0.9">h={h_val}</text>')

        # Floating Badge showing g and f values if known
        if known_g is not None:
            f_val = known_g + h_val
            badge_y = y - 44
            badge_text = f"g={known_g} | f={f_val}"
            badge_w = 76
            svg.append(f'<rect x="{x - badge_w/2}" y="{badge_y}" width="{badge_w}" height="20" rx="10" fill="#030712" stroke="{border_color}" stroke-width="1.5"/>')
            svg.append(f'<text x="{x}" y="{badge_y + 14}" fill="#f8fafc" font-size="10.5" font-weight="bold" font-family="monospace" text-anchor="middle">{badge_text}</text>')

    svg.append('</svg>')
    return "".join(svg)

# ============================ 4. APPLICATION UI LAYOUT ============================

# Sidebar Controls
st.sidebar.title("⭐ A* Tree Visualizer")
st.sidebar.caption("AI and Soft Computing (AISC) — Step-by-Step Project")
st.sidebar.divider()

# Session State for Playback
if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

# Speed Slider
speed = st.sidebar.slider("⏱️ Animation Speed (seconds)", min_value=0.3, max_value=2.5, value=1.0, step=0.1)

# Action Buttons
col_b1, col_b2, col_b3 = st.sidebar.columns(3)
if col_b1.button("⏮ Prev", use_container_width=True):
    st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
    st.session_state.playing = False

if col_b2.button("Next ⏭", use_container_width=True):
    st.session_state.step_idx = min(TOTAL_STEPS, st.session_state.step_idx + 1)
    st.session_state.playing = False

if col_b3.button("🔄 Reset", use_container_width=True):
    st.session_state.step_idx = 0
    st.session_state.playing = False

auto_btn_label = "⏸ Pause" if st.session_state.playing else "▶ Auto Play"
if st.sidebar.button(auto_btn_label, use_container_width=True, type="primary"):
    st.session_state.playing = not st.session_state.playing

# Slider to scrub directly to any point
selected_step = st.sidebar.slider(
    "Scrub Timeline",
    min_value=0,
    max_value=TOTAL_STEPS,
    value=st.session_state.step_idx,
    help="Drag to jump to any A* event instantly"
)

if selected_step != st.session_state.step_idx:
    st.session_state.step_idx = selected_step
    st.session_state.playing = False

# Legend in Sidebar
st.sidebar.divider()
st.sidebar.markdown("### 🎨 Color Legend")
st.sidebar.markdown("""
- 🟡 **Gold**: Final Optimal Path ($S \\to A \\to D \\to G$)
- 🟠 **Orange**: Current Selected / Focus Node
- 🔴 **Red**: Explored (`CLOSED` set)
- 🔵 **Blue**: Frontier (`OPEN` priority queue)
- 🟣 **Purple**: Goal Discovered
- ⚪ **Dark Slate**: Unvisited
""")

# Main Content Tabs
tab_viz, tab_table = st.tabs(["🚀 Interactive Visualizer", "📊 Dry Run Table"])

snap = SNAPSHOTS[st.session_state.step_idx]

with tab_viz:
    # Header Title
    st.markdown(f"## {snap['title']}")
    
    # Top Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Focus Node", snap["metrics"]["node"])
    m2.metric("Path Cost: g(n)", snap["metrics"]["g"])
    m3.metric("Heuristic: h(n)", snap["metrics"]["h"])
    m4.metric("Priority: f(n) = g + h", snap["metrics"]["f"])

    # Two Column Layout: Enlarged Graph on Left (65%), Data Panels on Right (35%)
    col_tree, col_panels = st.columns([1.9, 1.1], gap="medium")

    with col_tree:
        # Render enlarged hardware-accelerated SVG via iframe component to prevent ANY markdown escaping/code blocks
        svg_content = render_svg_tree(snap)
        html_wrapper = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <style>
            body, html {{
              margin: 0;
              padding: 0;
              width: 100%;
              height: 100%;
              overflow: hidden;
              background-color: transparent;
              display: flex;
              align-items: center;
              justify-content: center;
            }}
          </style>
        </head>
        <body>
          {svg_content}
        </body>
        </html>
        """
        components.html(html_wrapper, height=560, scrolling=False)
        st.caption(f"📍 Event {st.session_state.step_idx} of {TOTAL_STEPS} | Start: S | Goal: G | Optimal Cost = 9")

    with col_panels:
        # Explanation Card
        st.markdown(f"""
        <div class="status-card">
            <div class="status-title">💡 Algorithm Explanation</div>
            <div style="color: #cbd5e1; font-size: 0.94rem; line-height: 1.5;">{snap['expl']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Calculations Box
        calc_html = "".join([f'<div class="calc-row">{line}</div>' for line in snap["calc_lines"]])
        st.markdown(f"""
        <div class="status-card">
            <div class="status-title">🧮 Live Calculation</div>
            {calc_html}
        </div>
        """, unsafe_allow_html=True)

        # Live Priority Queue OPEN
        active_chips = "".join([f'<span class="queue-chip">{n} (f={f})</span>' for f, n in snap["open_valid"]]) if snap["open_valid"] else '<span style="color:#64748b;">(Empty)</span>'
        stale_chips = f'<div style="margin-top:6px;"><span style="color:#f87171; font-size:0.8rem; font-weight:bold;">⚠️ Ignored Old Entries:</span> ' + "".join([f'<span class="stale-chip">{n} (f={f})</span>' for f, n in snap["open_stale"]]) + '</div>' if snap["open_stale"] else ''
        st.markdown(f"""
        <div class="status-card">
            <div class="status-title">📥 OPEN Priority Queue (heapq)</div>
            <div>{active_chips}</div>
            {stale_chips}
        </div>
        """, unsafe_allow_html=True)

        # Live CLOSED Set
        closed_chips = " → ".join([f'<span class="closed-chip">{n}</span>' for n in snap["order"]]) if snap["order"] else '<span style="color:#64748b;">(None)</span>'
        st.markdown(f"""
        <div class="status-card">
            <div class="status-title">🔒 CLOSED Set (Exploration Order)</div>
            <div>{closed_chips}</div>
        </div>
        """, unsafe_allow_html=True)

# Dry Run Tab
with tab_table:
    st.markdown("### 📋 Complete A* Dry Run Walkthrough")
    st.markdown("""
| Step | Node Selected | g | h | f | Neighbours Processed | OPEN After Step | CLOSED After Step |
|---|---|---|---|---|---|---|---|
| **0** | — (init) | — | — | — | push S | S(f=7) | — |
| **1** | **S** | 0 | 7 | 7 | A (g=2, f=7), B (g=3, f=7) | A(7), B(7) | S |
| **2** | **A** *(tie A < B)* | 2 | 5 | 7 | C (g=4, f=7), D (g=6, f=9) | B(7), C(7), D(9) | S, A |
| **3** | **B** *(tie B < C)* | 3 | 4 | 7 | E (g=5, f=8), F (g=8, f=11) | C(7), E(8), D(9), F(11) | S, A, B |
| **4** | **C** | 4 | 3 | 7 | H (g=7, f=11) | E(8), D(9), F(11), H(11) | S, A, B, C |
| **5** | **E** | 5 | 3 | 8 | G (g=10, f=10 via E) | D(9), G(10), F(11), H(11) | S, A, B, C, E |
| **6** | **D** | 6 | 3 | 9 | G (new g=9 < 10 → **Update**) | G(9), F(11), H(11) *(old G(10) ignored)* | S, A, B, C, E, D |
| **7** | **G** *(Goal Popped)* | 9 | 0 | 9 | **Goal Selected → STOP** | F(11), H(11) | S, A, B, C, E, D, G |
    """)
    st.success("✅ **Final Optimal Path:** `S → A → D → G` with total cost **9**.")


# Handle Auto-play looping
if st.session_state.playing:
    if st.session_state.step_idx < TOTAL_STEPS:
        time.sleep(speed)
        st.session_state.step_idx += 1
        st.rerun()
    else:
        st.session_state.playing = False
        st.rerun()
