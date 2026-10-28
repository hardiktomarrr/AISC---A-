# A* Search Tree Visualizer — Setup, Code Explanation and Viva Guide

Files:

- `astar_tree_visualizer.py`: the whole project (one file).
- `astar_search.mp4`: a Full-HD (1920×1080) video of the complete animation, made by the program itself. Keep it as a backup in case the laptop misbehaves during the viva.

---

## 1. Installation and running (macOS)

Open **Terminal** in the folder that contains the file.

```bash
python3 -m pip install matplotlib
python3 astar_tree_visualizer.py
```

A window opens showing the tree. Click **Start** once and the whole search plays by itself (about 65 seconds at speed 1.0). **Pause/Resume**, **Reset** and the **Speed** slider are optional.

If `pip install` says *"externally-managed-environment"* (common with Homebrew Python), use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib
python astar_tree_visualizer.py
```

**Other modes**

| Command | What it does |
|---|---|
| `python3 astar_tree_visualizer.py --text` | No graphics. Prints every real A* step automatically, 1.5 s apart. |
| `python3 astar_tree_visualizer.py --save` | Saves a Full-HD video, `astar_search.mp4`. Needs FFmpeg (`brew install ffmpeg`) and takes several minutes. |

**If no window appears:**

- Run the file from Terminal, not inside a Jupyter notebook. In Jupyter, put `%matplotlib qt` or `%matplotlib osx` in the first cell.
- Force the native macOS backend: `MPLBACKEND=macosx python3 astar_tree_visualizer.py`.
- Last resort: use `--text` (timed automatic trace) or play the MP4 video. Both play with no clicking.

Everything works offline after Matplotlib is installed.

---

## 2. The sample problem

```
                    S (h=7)
               2 /         \ 3
           A (h=5)          B (h=4)
          2 /   \ 4        2 /   \ 5
      C (h=3)  D (h=3)  E (h=3)  F (h=3)
        3 |       3 \    / 5       | 2
      H (h=4)       G (h=0)      I (h=2)
```

- The edges are directed downward, parent → child.
- **G can be reached by two routes.** S→A→D→G costs 2+4+3 = **9**. S→B→E→G costs 3+2+5 = **10**.
- C→H and F→I are dead ends (they cannot reach G).
- **Admissible:** h(n) never overestimates the true cost. For example h(S)=7 ≤ 9, h(A)=5 ≤ 7, h(B)=4 ≤ 7, h(E)=3 ≤ 5 and h(D)=3 ≤ 3.
- **Consistent:** h(n) ≤ cost(n,m) + h(m) on every edge. For example h(S)=7 ≤ 2+h(A)=7 and h(D)=3 ≤ 3+h(G)=3.

The program checked both properties automatically during testing, along with a brute-force check that 9 is the minimum cost.

The example is designed to show three things a teacher likes to ask about:

1. **Ties** (A and B both have f=7, and later B and C both have f=7).
2. **Finding the goal through a worse route first** (G is reached through E with g=10).
3. **A better-path update** (D later finds G with g=9) and the outdated heap entry it leaves behind.

---

## 3. Function-by-function explanation

| Function | What it does (in simple words) |
|---|---|
| `define_graph()` | Returns three dictionaries: `graph` (who connects to whom, and at what cost), `heur` (the h value of each node), and `pos` (x, y coordinates used only for drawing). |
| `heuristic(n)` | Returns h(n), the guess of the remaining cost to G. |
| `astar_search(start, goal)` | Creates all the A* data structures (OPEN heap, CLOSED set, g, parent) and attaches the step generator. |
| `astar_steps(st, goal)` | **The real A* algorithm.** It is a *generator*: every `yield` pauses the algorithm and hands one event (for example "discovered C") to the animation. The next `next()` call resumes exactly where it stopped. So the animation never invents steps; it can only show what the algorithm just did. |
| `reconstruct_path(parent, node)` | Walks back from G using `parent` (G→D→A→S), then reverses the list to get S→A→D→G. |
| `open_entries(st)` | Reads the heap and separates valid entries from outdated duplicates, for display only. |
| `describe(ev, st, step)` | Turns an event into panel text (title, g/h/f calculations and an explanation), using live values from the data structures. |
| `queue_text(st)` | Formats the live OPEN list, ignored old heap entries and CLOSED list for the panel. |
| `setup_style()` | Picks a clean font that exists on the computer (Inter, Avenir Next or Helvetica). This is only for looks. |
| `styled_button(...)` | Creates one coloured, rounded Matplotlib button. |
| `node_color(n)` | Decides a node's colour by looking at the real structures. If the node is in `closed` it is red, if it is in `g` but not closed (so it is in OPEN) it is blue, and so on. |
| `draw_tree()` | Draws everything once: edges, cost labels, circles, h labels, legend, panel, buttons and slider. |
| `refresh_view()` | After each algorithm event, updates the existing objects (node colours and glows, g/f badges, arrow styles, panel text, status badge). Nothing is redrawn from scratch. |
| `marker_route(ev, st)` | Decides which edges the glowing marker travels along. When a node is selected, it travels down the real parent chain from S. When a neighbour is examined, it travels from the current node to that neighbour. |
| `advance()` | Calls `next()` on the generator to get the next real A* event, then refreshes the view. |
| `update_animation(frame)` | Called by `FuncAnimation` every 40 ms. It adds elapsed time, calls `advance()` when the current event's display time is over, moves the glowing marker and its fading trail, and pulses the ring and ripple around the current node. |
| `start_animation()`, `toggle_pause()`, `reset_visualization()` | Button handlers. Start is pressed once, and after that everything is automatic. |
| `text_mode()` | Prints the same events in the terminal with a timed pause (the fallback). |
| `save_video()` | Plays the same animation off-screen and writes a 1920×1080, 30 fps MP4. The buttons are hidden in the video. |

**About the code length:** the file is about 515 lines. Most of that is styling (the dark theme, glow effects, arrows and panel layout), the explanation sentences, comments and the optional text/video modes. The A* algorithm itself is about 35 lines (`astar_search` + `astar_steps` + `reconstruct_path`).

---

## 4. Data structures

| Structure | Python type | Example in this project | Purpose |
|---|---|---|---|
| Graph | `dict` of lists of `(neighbour, cost)` | `"A": [("C", 2), ("D", 4)]` | Adjacency list: fast lookup of a node's neighbours. |
| Heuristic table | `dict` | `{"S": 7, "A": 5, ...}` | h(n) for every node. |
| OPEN list | `list` used as a min-heap via `heapq` | `[(7, "A"), (7, "B")]` | Nodes discovered but not yet expanded. `heappop` always returns the **smallest f**, with ties broken by the node name. |
| CLOSED set | `set` | `{"S", "A"}` | Nodes already expanded. Membership checks are O(1). |
| `order` | `list` | `["S", "A", "B"]` | Same as CLOSED, but remembers the order (for display). |
| `g` | `dict` | `{"S": 0, "A": 2}` | Best known actual cost from S to each node. |
| `parent` | `dict` | `{"A": "S", "D": "A"}` | Where each node was best reached from. Used to rebuild the path. |

**Duplicate heap entries:** `heapq` cannot change the priority of an item already in the heap. So when a cheaper route is found, the program simply **pushes a new entry** (for example `(9, "G")`) and leaves the old one (`(10, "G")`) in the heap. When an old entry is popped later, the node is already in CLOSED, so the code skips it (`if n in closed: continue`). This is called *lazy deletion*.

---

## 5. A* pseudocode

```
A*(start, goal):
    g[start] = 0 ; parent[start] = None
    OPEN = priority queue containing (h(start), start)
    CLOSED = empty set
    while OPEN is not empty:
        (f, n) = pop node with lowest f from OPEN
        if n in CLOSED: continue                 # outdated entry
        add n to CLOSED
        if n == goal: return reconstruct_path(parent, goal)
        for each (m, cost) in neighbours(n):
            if m in CLOSED: continue
            new_g = g[n] + cost
            if new_g < g[m] (or m never seen):
                g[m] = new_g ; parent[m] = n
                push (new_g + h(m), m) into OPEN
    return FAILURE
```

---

## 6. Dry run (matches the program output exactly)

| Step | Node expanded | g | h | f | Neighbours processed | OPEN after step (f) | CLOSED after step |
|---|---|---|---|---|---|---|---|
| 0 | — (init) | — | — | — | push S | S(7) | — |
| 1 | S | 0 | 7 | 7 | A: g=0+2=2, f=7 · B: g=0+3=3, f=7 | A(7), B(7) | S |
| 2 | A (tie with B, A < B) | 2 | 5 | 7 | C: g=4, f=7 · D: g=6, f=9 | B(7), C(7), D(9) | S, A |
| 3 | B (tie with C, B < C) | 3 | 4 | 7 | E: g=5, f=8 · F: g=3+5=8, f=11 | C(7), E(8), D(9), F(11) | S, A, B |
| 4 | C | 4 | 3 | 7 | H: g=7, f=11 | E(8), D(9), F(11), H(11) | S, A, B, C |
| 5 | E | 5 | 3 | 8 | G: g=5+5=10, f=10 (parent E) | D(9), G(10), F(11), H(11) | S, A, B, C, E |
| 6 | D | 6 | 3 | 9 | G: new g=6+3=9 < 10 → **update**, parent(G)=D, push G(9) | G(9), F(11), H(11) + old G(10) ignored | S, A, B, C, E, D |
| 7 | **G selected → STOP** | 9 | 0 | 9 | — | F(11), H(11) | S, A, B, C, E, D, G |
**Path reconstruction:** parent(G)=D, parent(D)=A, parent(A)=S, so the path is **S → A → D → G** with cost 2 + 4 + 3 = **9**. The program expanded 7 nodes (including G).

**Final colours:** S, A, D, G are yellow (the path). B, C, E are red (explored, not on the path). F, H are blue (still waiting in OPEN, never needed). I is grey (never reached).

**Key observation for the viva:** at step 5 the goal G was already *discovered* with cost 10, but A* did **not** stop. It stops only when G is *selected* (popped with the lowest f). That rule is what let it find the cheaper route through D.

---

## 7. How the animation corresponds to the code

| What you see | Code that causes it |
|---|---|
| Pressing Start begins the search | `start_animation()` sets `running=True` and calls `advance()`, which runs the generator up to its first `yield ("init", ...)` |
| A new event every 1.5–3.5 s | `update_animation()` accumulates time. When `t ≥ DURATION[event]` it calls `next()` on the algorithm. |
| Orange node with a pulsing ring and ripple | The node just returned by `heapq.heappop` (events `select`/`discover`/`update`) |
| Orange glowing marker travelling down the arrows from S | `reconstruct_path(parent, n)`: the real parent chain to the selected node |
| Blue marker moving to a child | The `for nb, cost in GRAPH[n]` loop examining one neighbour |
| Node turns blue and shows g/f | `g[nb] = new_g` and `heappush(...)`: the node is now in OPEN |
| Node turns red | The node is in `closed` (it was expanded) |
| Goal turns purple | G has a g value (it has been discovered) |
| Bright arrow moves from E→G to D→G | `parent["G"]` changed from E to D (the better-path update) |
| "old heap entries (ignored): G(f=10)" | The outdated duplicate still in the heap |
| Glowing yellow arrows appearing one by one | `("path", ...)` events yielded after `reconstruct_path` |
| Panel numbers | `describe()` reads `g`, `heuristic()`, the heap and `closed` at that exact paused moment |

---

## 8. Likely viva questions

1. **What is A\*?**
   An informed best-first search that expands the node with the smallest f(n) = g(n) + h(n).

2. **What are g, h and f?**
   g = actual cost from the start to n. h = estimated cost from n to the goal. f = estimated total cost of the best path through n.

3. **What is an admissible heuristic?**
   One that never overestimates the true remaining cost: h(n) ≤ h\*(n).

4. **Why does admissibility guarantee optimality?**
   When G is popped with f = C, every other OPEN node has f ≥ C. Because h underestimates, none of them can lead to a path cheaper than C.

5. **What is a consistent (monotone) heuristic?**
   h(n) ≤ cost(n,m) + h(m) for every edge. It implies admissibility. It also means a node's g is already optimal when the node is expanded, so closed nodes never need reopening.

6. **Why use a priority queue / `heapq`?**
   To get the lowest-f node in O(log n) instead of scanning the whole list each time.

7. **Why does the heap store `(f, node)` tuples?**
   Python compares tuples element by element: f first, then the node name, which breaks ties.

8. **What are the OPEN and CLOSED lists?**
   OPEN holds nodes discovered but not expanded (the frontier). CLOSED holds nodes already expanded.

9. **Why didn't A\* stop when G was first discovered at step 5?**
   The goal test happens on selection, not discovery. A cheaper path (through D, f = 9) was still possible, and it was.

10. **What happens when a better path to a node is found?**
    Update g and parent, then push a new heap entry. The old entry is skipped later (lazy deletion).

11. **What is the parent dictionary for?**
    To rebuild the final path backwards from the goal.

12. **What if h(n) = 0 for every node?**
    A\* becomes Uniform Cost Search (Dijkstra). It is still optimal, but it explores more nodes.

13. **What if h overestimates?**
    A\* may return a non-optimal path.

14. **How does A\* differ from Greedy Best-First Search?**
    Greedy uses only h(n), so it is fast but not optimal. A\* adds g(n), so it accounts for the cost already paid.

15. **Is A\* complete?**
    Yes, on finite graphs (or with a finite branching factor and positive edge costs).

16. **Why are F and H still blue at the end?**
    Their f = 11 is higher than the optimal cost 9, so A\* never needed to expand them. This is how the heuristic saves work.

17. **Why is I grey?**
    Its parent F was never expanded, so I was never generated.

18. **Is the animation pre-recorded or hardcoded?**
    No. The algorithm is a Python generator. The animation calls `next()` and draws whatever the algorithm actually did. Change a cost or an h value and the animation changes too.

19. **Give real-world uses of A\*.**
    GPS route planning, game character path-finding, robot navigation and puzzle solving (8-puzzle).

20. **Where does a heuristic come from in practice?**
    From a relaxed version of the problem, such as straight-line distance on a map, or Manhattan distance on a grid.

---

## 9. Time and space complexity

- **b** = branching factor, **d** = depth of the optimal solution.
- **Worst-case time: O(b^d).** If the heuristic is uninformative (for example h = 0), A\* has to look at almost every node up to depth d. The number of nodes grows exponentially with depth (1 + b + b² + … + b^d). Each heap operation adds a log factor.
- **Worst-case space: O(b^d).** A\* keeps every generated node in OPEN or CLOSED. Memory, not time, is usually its practical limit.
- On an explicit graph with V nodes and E edges and a consistent heuristic, each node is expanded at most once: **O((V + E) log V)** time and **O(V)** space.
- **The better the heuristic** (the closer to the true cost, without overestimating), the fewer nodes A\* expands. A perfect heuristic expands only the nodes on the optimal path.

---

## 10. Informed vs uninformed search

- **Uninformed (blind) search**, such as BFS, DFS and Uniform Cost Search, uses only the problem definition: the start, the actions, the costs and the goal test. It has no idea which direction the goal is in.
- **Informed (heuristic) search**, such as Greedy Best-First and A\*, also uses domain knowledge h(n) to estimate how close each node is to the goal. This lets it focus on promising nodes.
- **A\* combines the two:** g(n) from Uniform Cost Search (the cost already paid, which guarantees optimality) and h(n) from greedy search (the estimate ahead, which gives speed).
- In this project, A\* never expanded F, H or I. A blind search such as BFS would have expanded them.
