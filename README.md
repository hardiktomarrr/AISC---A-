# A* Search Algorithm — Interactive Tree Visualizer

An interactive, step-by-step tree visualizer and learning tool for the **A\* (A-Star) search algorithm**, built for **AI & Soft Computing (AISC)**.

---

## 🌟 Features

- **Interactive Streamlit Web Visualizer (`app.py`)**:
  - Smooth, hardware-accelerated vector SVG tree representation.
  - Interactive playback: **Scrub Timeline** slider, **Next / Prev** steps, **Reset**, and **Auto Play** with variable speed.
  - Real-time priority calculations: $f(n) = g(n) + h(n)$.
  - Live inspection of the **OPEN priority queue** (with lazy deletion detection) and **CLOSED set**.
  - Built-in **Dry Run Table** and **Viva / Theory Examination Guide**.
- **Matplotlib Animated Visualizer (`astar_tree_visualizer.py`)**:
  - Standalone GUI visualizer using Matplotlib animations.
  - Supports `--text` terminal trace mode and `--save` high-definition MP4 export mode.
- **Reference Video (`astar_search.mp4`)**:
  - Pre-rendered 1080p demonstration video of the full search execution.

---

## 🌳 The Search Problem

The algorithm searches for the optimal path from start node **`S`** to goal node **`G`**:

```text
                    S (h=7)
               2 /         \ 3
           A (h=5)          B (h=4)
          2 /   \ 4        2 /   \ 5
      C (h=3)  D (h=3)  E (h=3)  F (h=3)
        3 |       3 \    / 5       | 2
      H (h=4)       G (h=0)      I (h=2)
```

- **Optimal Path:** `S → A → D → G`
- **Total Cost:** $2 + 4 + 3 = 9$
- **Heuristic Properties:** Admissible ($h(n) \le h^*(n)$) and Consistent ($h(n) \le c(n,m) + h(m)$).
- **Core Concepts Illustrated:**
  - Tie-breaking ($f(A) = f(B) = 7$, resolved alphabetically).
  - Suboptimal goal discovery ($G$ reached first via $E$ with cost 10) vs. goal selection.
  - Cheaper path update ($D$ finds $G$ with cost 9) and lazy deletion of outdated heap entries.

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone <YOUR_REPO_URL>
cd <REPO_FOLDER>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Streamlit Web Visualizer (Recommended)

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Run the Matplotlib Desktop Visualizer

```bash
# Interactive GUI
python3 astar_tree_visualizer.py

# Terminal text-only trace
python3 astar_tree_visualizer.py --text

# Export to 1080p MP4 (requires ffmpeg)
python3 astar_tree_visualizer.py --save
```

---

## 📚 Study & Viva Guide

See [`VIVA_GUIDE.md`](./VIVA_GUIDE.md) for dry-run tables, algorithm complexity analysis, and answers to 20 common viva questions.
