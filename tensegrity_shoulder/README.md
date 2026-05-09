# Tensegrity Shoulder Joint

A self-contained MuJoCo simulation of a single tensegrity shoulder joint
actuated by two velocity-controlled motors driving crank disks. The joint is
also exposed as a **graph**:

- **Nodes** = struts (capsule rods).
- **Edges** = cables (spatial tendons connecting two struts).

Every 50 simulation steps, the runner samples:

1. The world-frame `(x, y, z)` position of every node, and
2. The current stiffness of every tendon edge,

and saves them to `output/sample_log.npz`.

## Layout

```
tensegrity_shoulder/
  pyproject.toml
  .python-version
  model/
    scene.xml          # ground plane + lights, includes shoulder.xml
    shoulder.xml       # the single tensegrity shoulder joint + motors
    assets/wing.stl
  src/
    __init__.py
    graph.py           # builds the strut/cable graph from the MuJoCo model
    simulate.py        # runs the sim and logs samples every 50 steps
  output/              # NPZ logs land here
```

## Setup (uv)

```bash
cd tensegrity_shoulder
uv sync                  # creates .venv and installs everything
```

This produces a fully self-contained `.venv` inside the directory; you can move
the entire `tensegrity_shoulder/` folder anywhere and re-run `uv sync` to
re-materialize the environment, or copy the existing `.venv` along with it.

## Run

The MuJoCo passive viewer requires `mjpython` on macOS:

```bash
uv run mjpython -m src.simulate                 # interactive viewer
uv run mjpython -m src.simulate --duration 4.0  # auto-stop after 4 sim seconds
uv run python    -m src.simulate --headless --duration 5.0   # no viewer
```

The first run prints the discovered graph (node and edge listing) and then
starts simulating. While running, the controller drives both motor actuators
with a 1 Hz sinusoidal velocity command (the two crank disks are constrained
to rotate in opposite phase), causing the tensegrity to flex.

## Output format

`output/sample_log.npz` contains:

| key                | shape           | description                                   |
| ------------------ | --------------- | --------------------------------------------- |
| `times`            | `(T,)`          | Sim time (s) at each sample                   |
| `step_indices`     | `(T,)`          | Integration step index at each sample (× 50)  |
| `node_positions`   | `(T, N, 3)`     | World `(x, y, z)` of each node                |
| `tendon_stiffness` | `(T, E)`        | Stiffness of each cable                       |
| `node_names`       | `(N,)` strings  | Strut geom names                              |
| `edge_names`       | `(E,)` strings  | Cable tendon names                            |
| `edge_endpoints`   | `(E, 2)` int32  | `(node_a, node_b)` indices for each edge      |

`T` = number of samples, `N` = number of strut nodes, `E` = number of cable
edges.

Quick read-back:

```python
import numpy as np
log = np.load("output/sample_log.npz")
print(log["node_positions"].shape, log["tendon_stiffness"].shape)
```

## Spectral analysis

After building the graph we compute the combinatorial Laplacian `L = D - A`
weighted by per-cable stiffness, then take its eigendecomposition.
The lowest eigenvectors give a basis for graph-smooth signals.

We also experiment with **graph Fourier transform** denoising of noisy
position signals using a low-pass cutoff on the Laplacian spectrum.
