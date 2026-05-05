"""Graph representation of the tensegrity shoulder joint.

Each *tendon attachment site* is a node. The rods themselves are not nodes -
they are part of the rigid body that carries the sites. Each spatial tendon
(structural cable or motor cable) is an edge connecting the two endpoint sites
it wires together.

A node's (x, y, z) position at any sim step is the world-frame position of its
site, read directly from MuJoCo's ``data.site_xpos``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class SiteNode:
    """A single tendon attachment site treated as one graph node."""

    node_id: int
    site_name: str
    site_id: int
    body_id: int
    body_name: str

    def world_position(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        del model  # site_xpos is enough
        return np.array(data.site_xpos[self.site_id], dtype=np.float64)


@dataclass
class CableEdge:
    """A single cable (spatial tendon) connecting two site-nodes."""

    edge_id: int
    tendon_name: str
    tendon_id: int
    node_a: int
    node_b: int
    site_a_name: str
    site_b_name: str
    is_motor: bool


@dataclass
class TensegrityGraph:
    nodes: list[SiteNode]
    edges: list[CableEdge]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def sample_node_positions(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> np.ndarray:
        """Returns array of shape (num_nodes, 3) with world-frame node positions."""
        out = np.empty((self.num_nodes, 3), dtype=np.float64)
        for i, node in enumerate(self.nodes):
            out[i] = node.world_position(model, data)
        return out

    def sample_tendon_stiffness(self, model: mujoco.MjModel) -> np.ndarray:
        out = np.empty(self.num_edges, dtype=np.float64)
        for i, edge in enumerate(self.edges):
            out[i] = float(model.tendon_stiffness[edge.tendon_id])
        return out


STRUCTURAL_BODY_NAMES: tuple[str, ...] = ("part_1", "part_2")


def build_graph(model: mujoco.MjModel) -> TensegrityGraph:
    """Build the site/cable graph by introspecting the MuJoCo model.

    Nodes are tendon attachment sites that live on a *structural* body of the
    tensegrity (``part_1`` or ``part_2``). Sites on the motor crank disks are
    actuation inputs, not graph nodes, so they are excluded - and motor cables
    (which connect a strut site to a disk site) are excluded from the edge
    list as well, since they have only one structural endpoint.
    """

    # 1. Walk every spatial tendon, recording its endpoint site ids.
    tendon_site_ids: list[tuple[int, list[int]]] = []
    for tendon_id in range(model.ntendon):
        adr = int(model.tendon_adr[tendon_id])
        num = int(model.tendon_num[tendon_id])
        sids: list[int] = []
        for k in range(num):
            wtype = int(model.wrap_type[adr + k])
            if wtype == int(mujoco.mjtWrap.mjWRAP_SITE):
                sids.append(int(model.wrap_objid[adr + k]))
        if len(sids) >= 2:
            tendon_site_ids.append((tendon_id, sids))

    def _is_structural_site(site_id: int) -> bool:
        body_id = int(model.site_bodyid[site_id])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        return bname in STRUCTURAL_BODY_NAMES

    # 2. Collect referenced *structural* sites only.
    referenced: set[int] = set()
    for _, sids in tendon_site_ids:
        for sid in sids:
            if _is_structural_site(sid):
                referenced.add(sid)

    # 3. Build a node per structural site, in deterministic site_id order.
    nodes: list[SiteNode] = []
    site_id_to_node: dict[int, int] = {}
    for site_id in sorted(referenced):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        body_id = int(model.site_bodyid[site_id])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        node = SiteNode(
            node_id=len(nodes),
            site_name=sname,
            site_id=site_id,
            body_id=body_id,
            body_name=bname,
        )
        site_id_to_node[site_id] = node.node_id
        nodes.append(node)

    # 4. Build an edge per spatial tendon whose *both* endpoints are structural
    # nodes. Motor cables have a disk endpoint, so they are skipped here.
    edges: list[CableEdge] = []
    for tendon_id, sids in tendon_site_ids:
        s_a, s_b = sids[0], sids[-1]
        if s_a not in site_id_to_node or s_b not in site_id_to_node:
            continue
        tname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id) or ""
        node_a = site_id_to_node[s_a]
        node_b = site_id_to_node[s_b]
        edges.append(
            CableEdge(
                edge_id=len(edges),
                tendon_name=tname,
                tendon_id=tendon_id,
                node_a=node_a,
                node_b=node_b,
                site_a_name=nodes[node_a].site_name,
                site_b_name=nodes[node_b].site_name,
                is_motor=tname.startswith("motor_"),
            )
        )

    return TensegrityGraph(nodes=nodes, edges=edges)


def describe_graph(graph: TensegrityGraph) -> str:
    lines = [
        f"Tensegrity graph: {graph.num_nodes} nodes (tendon sites), "
        f"{graph.num_edges} edges (cables)",
        "Nodes:",
    ]
    for node in graph.nodes:
        lines.append(
            f"  [{node.node_id:2d}] {node.site_name:24s} body={node.body_name}"
        )
    lines.append("Edges:")
    for edge in graph.edges:
        kind = "motor " if edge.is_motor else "struct"
        lines.append(
            f"  [{edge.edge_id:2d}] {kind} {edge.tendon_name:24s} "
            f"{edge.site_a_name} <-> {edge.site_b_name}"
        )
    return "\n".join(lines)
