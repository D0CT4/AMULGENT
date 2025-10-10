#!/usr/bin/env python3
"""
Demo: Workflow Visualization for AMULGENT

Generates a visual graph of the multi-agent workflow and data flow using networkx
and matplotlib (fallback to graphviz if available). The visualization highlights:

- Agents and their roles
- Inter-agent communication channels
- HRM reasoning tiers and escalation paths
- Data sources, transformations, and sinks
- Security checkpoints and audit trail nodes

Usage:
    python demo_workflow_visualization.py --format png --output workflow.png

Notes:
- This demo is read-only; it does not require credentials or external APIs
- The graph structure is inferred from the canonical architecture documented in README
- If graphviz is installed, a DOT export is also produced for higher-quality layouts

Fixes: #2
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import networkx as nx
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "networkx is required for visualization. Install with `pip install networkx`."
    ) from e

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Node:
    """Graph node descriptor.

    Attributes:
        key: Unique node key
        label: Human-readable label
        kind: One of {'agent','core','hrm','data','security','io'}
    """

    key: str
    label: str
    kind: str


def _add_typed_node(g: nx.DiGraph, node: Node) -> None:
    """Add a typed node with style metadata to the graph."""
    palette = {
        "agent": {"color": "#1f77b4", "shape": "o"},
        "core": {"color": "#2ca02c", "shape": "s"},
        "hrm": {"color": "#9467bd", "shape": "D"},
        "data": {"color": "#17becf", "shape": "^"},
        "security": {"color": "#d62728", "shape": ">"},
        "io": {"color": "#ff7f0e", "shape": "v"},
    }
    style = palette.get(node.kind, palette["core"])
    g.add_node(
        node.key,
        label=node.label,
        kind=node.kind,
        color=style["color"],
        shape=style["shape"],
    )


def build_workflow_graph() -> nx.DiGraph:
    """Build the canonical AMULGENT workflow graph.

    Returns:
        Directed graph describing agents, core components, HRM tiers, and data flow.
    """
    g = nx.DiGraph()

    # Core components
    core_nodes = [
        Node("system", "AIMultiAgentSystem", "core"),
        Node("coord", "Coordinator", "core"),
        Node("config", "Config", "core"),
    ]

    # Agents
    agent_nodes = [
        Node("agent_base", "BaseAgent", "agent"),
        Node("agent_analysis", "AnalysisAgent", "agent"),
        Node("agent_hrm", "HRMReasoning", "agent"),
    ]

    # HRM tiers
    hrm_nodes = [
        Node("hrm_t1", "Tier 1: Pattern", "hrm"),
        Node("hrm_t2", "Tier 2: Heuristic", "hrm"),
        Node("hrm_t3", "Tier 3: Deep Analysis", "hrm"),
        Node("hrm_t4", "Tier 4: Full LLM", "hrm"),
    ]

    # Data and IO
    data_nodes = [
        Node("io_in", "Input", "io"),
        Node("data_norm", "Normalization", "data"),
        Node("data_cache", "Cache", "data"),
        Node("io_out", "Output", "io"),
    ]

    # Security and audit
    sec_nodes = [
        Node("sec_validation", "Validation", "security"),
        Node("sec_policy", "Policy/ACL", "security"),
        Node("sec_audit", "Audit Trail", "security"),
    ]

    for n in core_nodes + agent_nodes + hrm_nodes + data_nodes + sec_nodes:
        _add_typed_node(g, n)

    # Core wiring
    g.add_edge("system", "coord", label="init")
    g.add_edge("system", "config", label="reads")
    g.add_edge("coord", "agent_analysis", label="dispatch")
    g.add_edge("coord", "agent_hrm", label="dispatch")

    # Data flow
    g.add_edge("io_in", "sec_validation", label="sanitize")
    g.add_edge("sec_validation", "data_norm", label="normalize")
    g.add_edge("data_norm", "agent_analysis", label="features")
    g.add_edge("data_norm", "agent_hrm", label="context")
    g.add_edge("agent_analysis", "data_cache", label="store")
    g.add_edge("agent_hrm", "data_cache", label="store")
    g.add_edge("data_cache", "io_out", label="serve")

    # Security and audit
    g.add_edge("coord", "sec_policy", label="check")
    g.add_edge("sec_policy", "coord", label="allow/deny")
    g.add_edge("system", "sec_audit", label="log")

    # HRM escalation path
    g.add_edge("agent_hrm", "hrm_t1", label="start")
    g.add_edge("hrm_t1", "hrm_t2", label="escalate if needed")
    g.add_edge("hrm_t2", "hrm_t3", label="escalate if needed")
    g.add_edge("hrm_t3", "hrm_t4", label="escalate if needed")
    g.add_edge("hrm_t1", "data_cache", label="result")
    g.add_edge("hrm_t2", "data_cache", label="result")
    g.add_edge("hrm_t3", "data_cache", label="result")
    g.add_edge("hrm_t4", "data_cache", label="result")

    return g


def _draw_with_matplotlib(g: nx.DiGraph, output: str) -> None:
    """Draw graph using matplotlib with type-aware styling."""
    pos = nx.spring_layout(g, seed=42)

    # Collect per-kind nodes
    kinds: Dict[str, List[str]] = {}
    for n, data in g.nodes(data=True):
        kinds.setdefault(data.get("kind", "core"), []).append(n)

    # Draw per-kind with different shapes/colors
    for kind, nodes in kinds.items():
        colors = [g.nodes[n].get("color", "#333333") for n in nodes]
        shapes = set(g.nodes[n].get("shape", "o") for n in nodes)
        for shape in shapes:
            shaped_nodes = [n for n in nodes if g.nodes[n].get("shape") == shape]
            nx.draw_networkx_nodes(
                g,
                pos,
                nodelist=shaped_nodes,
                node_color=[g.nodes[n]["color"] for n in shaped_nodes],
                node_shape=shape,
                node_size=1500,
                alpha=0.9,
            )

    nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle="-|>", width=1.6, alpha=0.6)

    labels = {n: g.nodes[n].get("label", n) for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)

    edge_labels = {(u, v): d.get("label", "") for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=7)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(f"Saved workflow visualization to: {output}")


def _export_dot(g: nx.DiGraph, path: str) -> None:
    """Export DOT file for Graphviz if available."""
    try:
        from networkx.drawing.nx_pydot import write_dot
    except Exception:  # pragma: no cover - optional dependency
        print("Graphviz export unavailable (install pydot). Skipping DOT export.")
        return

    write_dot(g, path)
    print(f"Exported DOT file to: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMULGENT workflow visualization demo")
    parser.add_argument(
        "--format",
        choices=["png", "svg"],
        default="png",
        help="Output image format",
    )
    parser.add_argument(
        "--output",
        default="workflow.png",
        help="Output file path (e.g., workflow.png)",
    )
    parser.add_argument(
        "--dot",
        action="store_true",
        help="Also export a DOT file for Graphviz layouts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Enforce extension
    root, ext = os.path.splitext(args.output)
    if args.format == "png" and ext.lower() not in {".png"}:
        args.output = f"{root}.png"
    if args.format == "svg" and ext.lower() not in {".svg"}:
        args.output = f"{root}.svg"

    g = build_workflow_graph()

    # Render
    _draw_with_matplotlib(g, args.output)

    # Optional DOT export
    if args.dot:
        _export_dot(g, f"{root}.dot")


if __name__ == "__main__":
    main()
