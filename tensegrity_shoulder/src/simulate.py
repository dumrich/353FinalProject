"""Simulate the single tensegrity shoulder joint and sample graph state.

Runs a MuJoCo passive viewer with the motor actuators driving the crank disks.
Every 50 simulation steps, samples the (x, y, z) position of every strut node
and the current stiffness of every cable edge, and appends them to NPZ logs.

Usage (must use mjpython on macOS for the interactive viewer):

    mjpython -m src.simulate
    mjpython -m src.simulate --headless --duration 5.0   # no viewer

Outputs are written to ./output/sample_log.npz next to this directory.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .graph import TensegrityGraph, build_graph, describe_graph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = PROJECT_ROOT / "model" / "scene.xml"
OUTPUT_DIR = PROJECT_ROOT / "output"

SAMPLE_INTERVAL_STEPS = 50
PRINT_INTERVAL_STEPS = 100


def visualize_graph(
    graph: "TensegrityGraph",
    model: mujoco.MjModel,
    data: mujoco.MjData,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Render the initial graph topology as a 3D plot.

    Nodes are colored by which structural body they live on; edges are colored
    and width-scaled by their tendon stiffness ("edge weight").
    """
    import threading

    import matplotlib

    # macOS / mjpython runs user code on a worker thread, but the default
    # 'macosx' backend can only open windows from the main thread. If we're
    # not on the main thread, force a non-interactive backend and save to disk.
    on_main_thread = threading.current_thread() is threading.main_thread()
    if not on_main_thread:
        matplotlib.use("Agg", force=True)
        if save_path is None:
            save_path = OUTPUT_DIR / "graph_topology.png"
        if show:
            print(
                "[visualize_graph] running off main thread (likely mjpython); "
                f"using Agg backend and saving to {save_path} instead of opening a window."
            )
            show = False

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    positions = graph.sample_node_positions(model, data)
    stiffness = graph.sample_tendon_stiffness(model)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    body_colors = {"part_1": "#1f77b4", "part_2": "#d62728"}
    default_color = "#7f7f7f"

    for node, pos in zip(graph.nodes, positions, strict=True):
        color = body_colors.get(node.body_name, default_color)
        ax.scatter(*pos, color=color, s=55, edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(
            pos[0], pos[1], pos[2] + 0.003,
            node.site_name.replace("part_", "p").replace("_t", "."),
            fontsize=7, color=color, ha="center",
        )

    if len(stiffness) > 0:
        norm = Normalize(vmin=float(stiffness.min()), vmax=float(stiffness.max() + 1e-9))
        cmap = cm.get_cmap("viridis")
    else:
        norm, cmap = None, None

    for edge in graph.edges:
        a = positions[edge.node_a]
        b = positions[edge.node_b]
        k = stiffness[edge.edge_id]
        if norm is not None:
            color = cmap(norm(k))
        else:
            color = "gray"
        lw = 0.8 + 2.5 * (norm(k) if norm is not None else 0.5)
        ls = "--" if edge.is_motor else "-"
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
            color=color, linewidth=lw, linestyle=ls, alpha=0.85, zorder=2,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(
        f"Tensegrity graph - {graph.num_nodes} nodes, {graph.num_edges} edges\n"
        "edge color/width = tendon stiffness (N/m)"
    )

    if cmap is not None and norm is not None:
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(stiffness)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.08)
        cbar.set_label("stiffness [N/m]")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=body_colors["part_1"],
                   markersize=8, label="part_1 (movable)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=body_colors["part_2"],
                   markersize=8, label="part_2 (base)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved graph visualization -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add zero-mean Gaussian noise to ``signal`` at the given SNR (in dB).

    SNR_dB = 10 * log10(signal_power / noise_power), where ``signal_power`` is
    the variance of the AC component (mean removed) so the SNR is well defined
    even for signals with a large DC offset like world-frame positions.
    """
    rng = rng or np.random.default_rng(0)
    ac = signal - np.mean(signal)
    sig_power = float(np.mean(ac ** 2))
    if sig_power <= 0.0:
        return signal.copy()
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def plot_noisy_signal(
    log_path: Path,
    node: str | int = 0,
    axis: str = "x",
    snr_db: float = 20.0,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot one node's 1D position trace with AWGN added at ``snr_db`` SNR."""
    import threading

    import matplotlib

    on_main_thread = threading.current_thread() is threading.main_thread()
    if not on_main_thread:
        matplotlib.use("Agg", force=True)
        if save_path is None:
            save_path = OUTPUT_DIR / "noisy_signal.png"
        if show:
            print(
                "[plot_noisy_signal] running off main thread; using Agg backend "
                f"and saving to {save_path} instead of opening a window."
            )
            show = False

    import matplotlib.pyplot as plt

    log = np.load(log_path, allow_pickle=False)
    times = log["times"]
    positions = log["node_positions"]  # (T, N, 3)
    names = [str(n) for n in log["node_names"].tolist()]

    if isinstance(node, str):
        if node not in names:
            raise ValueError(f"Node {node!r} not in log; available: {names}")
        node_idx = names.index(node)
    else:
        node_idx = int(node)
    node_name = names[node_idx]

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]

    clean = positions[:, node_idx, axis_idx]
    noisy = add_awgn(clean, snr_db=snr_db, rng=np.random.default_rng(0))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(times, clean, color="#1f77b4", linewidth=2.0, label="clean")
    ax.plot(times, noisy, color="#d62728", linewidth=0.9, alpha=0.8,
            label=f"noisy ({snr_db:.0f} dB SNR AWGN)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(f"{axis.lower()} position [m]")
    ax.set_title(f"Node '{node_name}' {axis.lower()}-position vs. time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved noisy-signal plot -> {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _tendon_tensions(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Approximate per-tendon spring tension (N).

    Tension = stiffness * max(0, length - rest_length). Cables can only pull,
    so we clamp negative slack to zero.
    """
    rest = np.asarray(model.tendon_lengthspring)[:, 0]  # lower bound = rest length
    length = np.asarray(data.ten_length)
    stiff = np.asarray(model.tendon_stiffness)
    return stiff * np.maximum(0.0, length - rest)


def _render_status(
    graph: "TensegrityGraph",
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step_count: int,
) -> str:
    positions = graph.sample_node_positions(model, data)
    tensions = _tendon_tensions(model, data)
    lines = [
        f"step {step_count:>7}   sim_time {data.time:7.3f}s",
        "",
        "Nodes (world x, y, z) [m]:",
        f"  {'id':>3}  {'name':<22}  {'x':>9}  {'y':>9}  {'z':>9}",
    ]
    for node, pos in zip(graph.nodes, positions, strict=True):
        lines.append(
            f"  {node.node_id:>3}  {node.site_name:<22}  "
            f"{pos[0]:>9.4f}  {pos[1]:>9.4f}  {pos[2]:>9.4f}"
        )
    lines.append("")
    lines.append("Cables (tension) [N]:")
    lines.append(f"  {'id':>3}  {'name':<22}  {'kind':<6}  {'tension':>9}")
    for edge in graph.edges:
        kind = "motor" if edge.is_motor else "struct"
        lines.append(
            f"  {edge.edge_id:>3}  {edge.tendon_name:<22}  {kind:<6}  "
            f"{tensions[edge.tendon_id]:>9.4f}"
        )
    return "\n".join(lines)


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 500) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)


def _control_signal(t: float) -> tuple[float, float]:
    """Constant motor velocity command so each crank disk rotates continuously.

    The equality constraint between the two hinges is ``q1 = pi + q4``, so the
    two disks must rotate in the *same* direction (with a constant 180-degree
    phase offset). Commanding opposite-sign velocities fights the constraint
    and pins the system. We therefore drive both motors with the same sign.
    """
    del t  # unused; kept for signature compatibility
    speed = 3.5  # rad/s (~0.56 rev/s)
    return speed, speed


def run(
    duration: float | None = None,
    headless: bool = False,
    output_path: Path | None = None,
    visualize_only: bool = False,
    visualize_save: Path | None = None,
) -> Path | None:
    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"Scene not found: {SCENE_PATH}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    graph = build_graph(model)
    print(describe_graph(graph))
    print()

    _settle(model, data, steps=200)

    if visualize_only:
        visualize_graph(graph, model, data, save_path=visualize_save, show=True)
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or (OUTPUT_DIR / "sample_log.npz")

    times: list[float] = []
    step_indices: list[int] = []
    node_positions: list[np.ndarray] = []  # each (N, 3)
    tendon_stiffness: list[np.ndarray] = []  # each (E,)

    def maybe_sample(step_idx: int) -> None:
        if step_idx % SAMPLE_INTERVAL_STEPS != 0:
            return
        times.append(float(data.time))
        step_indices.append(step_idx)
        node_positions.append(graph.sample_node_positions(model, data))
        tendon_stiffness.append(graph.sample_tendon_stiffness(model))

    def apply_control() -> None:
        u1, u2 = _control_signal(data.time)
        data.ctrl[0] = u1
        data.ctrl[1] = u2

    # ANSI: move cursor to home + clear screen below; rewritten each refresh.
    CLEAR = "\x1b[H\x1b[J"
    first_print = [True]

    def maybe_print(step_idx: int) -> None:
        if step_idx % PRINT_INTERVAL_STEPS != 0:
            return
        frame = _render_status(graph, model, data, step_idx)
        if first_print[0]:
            print("\x1b[2J", end="")  # full clear on first frame
            first_print[0] = False
        print(CLEAR + frame, end="", flush=True)

    step_count = 0

    if headless:
        end_time = (data.time + duration) if duration else (data.time + 5.0)
        while data.time < end_time:
            apply_control()
            mujoco.mj_step(model, data)
            step_count += 1
            maybe_sample(step_count)
            maybe_print(step_count)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall_start = time.time()
            sim_start = data.time
            end_time = (data.time + duration) if duration else None

            while viewer.is_running():
                if end_time is not None and data.time >= end_time:
                    break
                apply_control()
                mujoco.mj_step(model, data)
                step_count += 1
                maybe_sample(step_count)
                maybe_print(step_count)

                # Real-time pacing.
                dt = (data.time - sim_start) - (time.time() - wall_start)
                if dt > 0:
                    time.sleep(dt)
                if step_count % 16 == 0:
                    viewer.sync()

    if not node_positions:
        print("No samples recorded.")
        return output_path

    np.savez(
        output_path,
        times=np.array(times),
        step_indices=np.array(step_indices),
        node_positions=np.stack(node_positions, axis=0),  # (T, N, 3)
        tendon_stiffness=np.stack(tendon_stiffness, axis=0),  # (T, E)
        node_names=np.array([n.site_name for n in graph.nodes]),
        edge_names=np.array([e.tendon_name for e in graph.edges]),
        edge_endpoints=np.array(
            [(e.node_a, e.node_b) for e in graph.edges], dtype=np.int32
        ),
    )

    print(f"\nSamples: {len(times)}")
    print(
        f"node_positions shape: ({len(times)}, {graph.num_nodes}, 3) | "
        f"tendon_stiffness shape: ({len(times)}, {graph.num_edges})"
    )
    print(f"Saved log -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Sim seconds to run (default: indefinite with viewer, 5s headless).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to NPZ log (default: output/sample_log.npz).",
    )
    parser.add_argument(
        "--visualize-graph",
        action="store_true",
        help="Plot the initial graph topology (nodes + edges colored by stiffness) and exit.",
    )
    parser.add_argument(
        "--visualize-save",
        type=Path,
        default=None,
        help="If --visualize-graph is set, also save the figure to this path (PNG/SVG/PDF).",
    )
    parser.add_argument(
        "--plot-noisy-signal",
        action="store_true",
        help="After (or instead of) simulating, plot one node's 1D position with AWGN added.",
    )
    parser.add_argument(
        "--noise-node",
        default="0",
        help="Node to plot: integer index or site name (default: 0).",
    )
    parser.add_argument(
        "--noise-axis",
        choices=("x", "y", "z"),
        default="x",
        help="Position component to plot (default: x).",
    )
    parser.add_argument(
        "--noise-snr-db",
        type=float,
        default=20.0,
        help="Signal-to-noise ratio of the added AWGN (default: 20 dB).",
    )
    parser.add_argument(
        "--noise-save",
        type=Path,
        default=None,
        help="If set, save the noisy-signal figure to this path.",
    )
    args = parser.parse_args()

    if args.plot_noisy_signal:
        log_path = args.output or (OUTPUT_DIR / "sample_log.npz")
        if not log_path.exists():
            print(f"No log at {log_path}; running headless first to generate it...")
            run(
                duration=args.duration if args.duration is not None else 5.0,
                headless=True,
                output_path=log_path,
            )
        try:
            node_arg: str | int = int(args.noise_node)
        except ValueError:
            node_arg = args.noise_node
        plot_noisy_signal(
            log_path=log_path,
            node=node_arg,
            axis=args.noise_axis,
            snr_db=args.noise_snr_db,
            save_path=args.noise_save,
            show=True,
        )
        return

    run(
        duration=args.duration,
        headless=args.headless,
        output_path=args.output,
        visualize_only=args.visualize_graph,
        visualize_save=args.visualize_save,
    )


if __name__ == "__main__":
    main()
