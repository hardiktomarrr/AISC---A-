"""
Streamlit Web App: A* Search Tree Visualizer (Fast, Interactive & Zero-Lag)
AI and Soft Computing Academic Project
"""
import streamlit as st
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

# SVG coordinates (width=900, height=520)
NODE_COORDS = {
    "S": (450, 60),
    "A": (230, 180),
    "B": (670, 180),
    "C": (120, 310),
    "D": (340, 310),
    "E": (560, 310),
    "F": (780, 310),
    "H": (120, 450),
    "G": (450, 450),
    "I": (780, 450)
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
        "expl": "Press 'Next Step' or 'Auto Play' to begin. The start node S will be expanded first.",
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
                "expl": f"Node {n} was already closed with a cheaper route. This duplicate entry with f={f} is ignored (lazy deletion).",
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
                "expl": f"The goal {n} now has the lowest f={g[n]} in the priority queue. Since h(n) is admissible, no remaining path can be cheaper. A* stops!",
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

# ============================ 3. SVG TREE RENDERING ============================
def render_svg_tree(snap):
    """Generate crisp, hardware-accelerated SVG with modern styling."""
    curr_n = snap["current_node"]
    closed_set = snap["closed"]
    open_nodes = {n for f, n in snap["open_valid"]}
    path_edges = set(snap["path_edges"])
    active_edge = snap["active_edge"]
    g_dict = snap["g"]
    is_done = snap["event_kind"] == "done"
    
    # Path nodes
    path_nodes = set()
    if path_edges:
        for u, v in path_edges:
            path_nodes.add(u)
            path_nodes.add(v)
    elif is_done:
        path_nodes = {"S", "A", "D", "G"}

    svg_elements = []
    
    # SVG Definition & Styles
    svg_elements.append('''
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 520" style="width:100%; height:auto; background: #0e1117; border-radius: 12px; border: 1px solid #2d3748; box-shadow: 0 8px 30px rgba(0,0,0,0.5);">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#4a5568"/>
        </marker>
        <marker id="arrow-gold" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#ecc94b"/>
        </marker>
        <marker id="arrow-active" viewBox="0 0 10 10" refX="28" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#ed8936"/>
        </marker>
        <filter id="glow-gold" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
        <filter id="glow-orange" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="8" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>
    ''')

    # Draw Edges
    for u, neighbours in GRAPH.items():
        x1, y1 = NODE_COORDS[u]
        for v, cost in neighbours:
            x2, y2 = NODE_COORDS[v]
            
            # Check edge status
            is_path = (u, v) in path_edges or (is_done and (u, v) in [("S","A"), ("A","D"), ("D","G")])
            is_active = (active_edge == (u, v))
            
            if is_path:
                stroke_color = "#ecc94b"
                stroke_width = "4.5"
                marker = "url(#arrow-gold)"
                filter_attr = 'filter="url(#glow-gold)"'
            elif is_active:
                stroke_color = "#ed8936"
                stroke_width = "3.5"
                marker = "url(#arrow-active)"
                filter_attr = 'filter="url(#glow-orange)"'
            else:
                stroke_color = "#334155"
                stroke_width = "2"
                marker = "url(#arrow)"
                filter_attr = ""

            svg_elements.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke_color}" stroke-width="{stroke_width}" marker-end="{marker}" {filter_attr}/>'
            )
            
            # Edge Cost Label
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            # Offset slightly to not collide with line
            ox, oy = (my - y1) * 0.1, (x1 - mx) * 0.1
            svg_elements.append(f'''
            <rect x="{mx + ox - 12}" y="{my + oy - 10}" width="24" height="20" rx="4" fill="#1a202c" stroke="{stroke_color}" stroke-width="1"/>
            <text x="{mx + ox}" y="{my + oy + 4}" fill="#e2e8f0" font-size="12" font-weight="bold" font-family="sans-serif" text-anchor="middle">{cost}</text>
            ''')

    # Draw Nodes
    for node, (x, y) in NODE_COORDS.items():
        h_val = HEUR[node]
        known_g = g_dict.get(node)
        
        # Determine Node Theme Color
        if is_done and node in path_nodes:
            bg_color = "#d69e2e"
            border_color = "#ecc94b"
            text_color = "#000"
            glow = 'filter="url(#glow-gold)"'
        elif node == curr_n and snap["event_kind"] not in ("ready", "done"):
            bg_color = "#dd6b20"
            border_color = "#fbd38d"
            text_color = "#fff"
            glow = 'filter="url(#glow-orange)"'
        elif node in path_nodes:
            bg_color = "#d69e2e"
            border_color = "#ecc94b"
            text_color = "#000"
            glow = 'filter="url(#glow-gold)"'
        elif node in closed_set:
            bg_color = "#9b2c2c"
            border_color = "#fc8181"
            text_color = "#fff"
            glow = ""
        elif node in open_nodes:
            bg_color = "#2b6cb0"
            border_color = "#63b3ed"
            text_color = "#fff"
            glow = ""
        elif node == GOAL and known_g is not None:
            bg_color = "#6b46c1"
            border_color = "#b794f4"
            text_color = "#fff"
            glow = ""
        else:
            bg_color = "#1e293b"
            border_color = "#475569"
            text_color = "#cbd5e1"
            glow = ""

        # Outer Pulsing Ring for Active Node
        if node == curr_n and snap["event_kind"] not in ("ready", "done"):
            svg_elements.append(f'<circle cx="{x}" cy="{y}" r="32" fill="none" stroke="#ed8936" stroke-width="2" stroke-dasharray="4,4" opacity="0.8"/>')

        # Node Circle
        svg_elements.append(f'<circle cx="{x}" cy="{y}" r="24" fill="{bg_color}" stroke="{border_color}" stroke-width="3" {glow}/>')
        
        # Node Name & Heuristic inside circle
        svg_elements.append(f'''
        <text x="{x}" y="{y - 2}" fill="{text_color}" font-size="15" font-weight="900" font-family="sans-serif" text-anchor="middle">{node}</text>
        <text x="{x}" y="{y + 13}" fill="{text_color}" font-size="10" font-family="sans-serif" text-anchor="middle" opacity="0.85">h={h_val}</text>
        ''')

        # Floating Badge showing g and f values if known
        if known_g is not None:
            f_val = known_g + h_val
            badge_y = y - 36
            badge_text = f"g={known_g} | f={f_val}"
            badge_w = 66
            svg_elements.append(f'''
            <rect x="{x - badge_w/2}" y="{badge_y}" width="{badge_w}" height="17" rx="8" fill="#0f172a" stroke="{border_color}" stroke-width="1.2"/>
            <text x="{x}" y="{badge_y + 12}" fill="#f8fafc" font-size="9.5" font-weight="bold" font-family="monospace" text-anchor="middle">{badge_text}</text>
            ''')

    svg_elements.append('</svg>')
    return "".join(svg_elements)

# ============================ 4. APPLICATION UI LAYOUT ============================

# Sidebar Controls
st.sidebar.title("⭐ A* Tree Visualizer")
st.sidebar.markdown("**Interactive AI & Soft Computing Visualizer**")
st.sidebar.divider()

# Session State for Playback
if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

# Auto-play tick logic
speed = st.sidebar.slider("⏱️ Animation Speed (seconds)", min_value=0.3, max_value=2.5, value=1.0, step=0.1)

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
- ⚪ **Dark Grey**: Unvisited
""")

# Main Content Tabs
tab_viz, tab_table, tab_viva = st.tabs(["🚀 Interactive Visualizer", "📊 Dry Run Table", "🎓 Viva & Theory Guide"])

snap = SNAPSHOTS[st.session_state.step_idx]

with tab_viz:
    # Header Banner
    st.markdown(f"### {snap['title']}")
    
    # Top Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Focus Node", snap["metrics"]["node"])
    c2.metric("Path Cost: g(n)", snap["metrics"]["g"])
    c3.metric("Heuristic: h(n)", snap["metrics"]["h"])
    c4.metric("Total Priority: f(n)", snap["metrics"]["f"])

    # 2-Column Layout: Visualizer on Left, Live Queues on Right
    col_tree, col_queues = st.columns([1.7, 1.0])

    with col_tree:
        st.markdown(render_svg_tree(snap), unsafe_allow_html=True)
        st.caption(f"Step {st.session_state.step_idx} of {TOTAL_STEPS} — Hardware-accelerated SVG Rendering")

    with col_queues:
        # Explanation Card
        st.info(f"**Explanation:**\n\n{snap['expl']}")
        
        # Calculation breakdown
        with st.container(border=True):
            st.markdown("#### 🧮 Calculations")
            for line in snap["calc_lines"]:
                st.code(line, language="text")

        # Live Priority Queue OPEN
        with st.container(border=True):
            st.markdown("#### 📥 OPEN Priority Queue (`heapq`)")
            if snap["open_valid"]:
                valid_str = "  |  ".join([f"**{n}** (f={f})" for f, n in snap["open_valid"]])
                st.markdown(f"**Active Candidates:**\n\n{valid_str}")
            else:
                st.write("*(Empty)*")

            if snap["open_stale"]:
                stale_str = ", ".join([f"{n} (f={f})" for f, n in snap["open_stale"]])
                st.warning(f"⚠️ **Outdated Duplicates (Lazy Deletion):** {stale_str}")

        # Live CLOSED Set
        with st.container(border=True):
            st.markdown("#### 🔒 CLOSED Set (Explored)")
            if snap["order"]:
                st.markdown(" → ".join([f"`{n}`" for n in snap["order"]]))
            else:
                st.write("*(None)*")

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

# Viva Guide Tab
with tab_viva:
    st.markdown("### 🎓 Viva Preparation & Core Theory")
    
    with st.expander("1. What is the A* Search Algorithm?", expanded=True):
        st.write("""
        A* is an **informed (heuristic) best-first search algorithm** that finds the lowest-cost path from a start node to a goal node.
        At each step, it chooses to expand the node $n$ with the minimum evaluation function:
        $$f(n) = g(n) + h(n)$$
        - $g(n)$: The exact cost paid to reach node $n$ from the start.
        - $h(n)$: The estimated (heuristic) remaining cost from $n$ to the goal.
        - $f(n)$: The estimated total cost of the path passing through $n$.
        """)

    with st.expander("2. What are Admissibility and Consistency?"):
        st.write("""
        - **Admissible Heuristic:** $h(n) \\le h^*(n)$ for all nodes $n$. The heuristic never overestimates the true cost to reach the goal.
        - **Consistent (Monotone) Heuristic:** For every edge $(n, m)$ with cost $c(n, m)$:
          $$h(n) \\le c(n, m) + h(m)$$
          Consistency implies admissibility. With a consistent heuristic, the first time a node is expanded from `OPEN`, its $g$-value is guaranteed to be optimal. Thus, closed nodes never need to be reopened!
        """)

    with st.expander("3. Why did A* not stop when G was first discovered at Step 5?"):
        st.write("""
        At Step 5, goal $G$ was discovered through $E$ with cost $g(G) = 10$. However, **A* only checks the goal condition when a node is selected (popped) from OPEN**, not when it is generated/discovered.
        Because node $D$ had a lower priority ($f(D) = 9$), $D$ was popped next, which discovered the cheaper path to $G$ with cost $9$.
        """)

    with st.expander("4. How does Python's `heapq` handle decreasing key / duplicate entries?"):
        st.write("""
        Python's standard `heapq` module does not support an $O(\\log n)$ `decrease-key` operation.
        When a cheaper path to an existing node is found, the program pushes the new `(new_f, node)` tuple into the heap and keeps the old entry.
        When the old entry is eventually popped later, the code checks:
        `if n in closed: continue`
        and skips it. This is known as **lazy deletion**.
        """)

    with st.expander("5. Time and Space Complexity"):
        st.write("""
        - **Worst-case Time Complexity:** $O(b^d)$ where $b$ is the branching factor and $d$ is the depth of the optimal solution.
        - **Worst-case Space Complexity:** $O(b^d)$ because A* retains all generated nodes in memory (`OPEN` and `CLOSED`). Memory is the primary practical bottleneck for A*.
        """)

# Handle Auto-play looping
if st.session_state.playing:
    if st.session_state.step_idx < TOTAL_STEPS:
        time.sleep(speed)
        st.session_state.step_idx += 1
        st.rerun()
    else:
        st.session_state.playing = False
        st.rerun()
