"""Shared acyclicity check for parent/child edges, used by both graph types (S2.5, S3.5: the
same rule applies at the per-provider and reconciled levels).
"""

from __future__ import annotations


def assert_acyclic(child_edges: dict[str, tuple[str, ...]]) -> None:
    """Raises ValueError if the parent/child edges (keyed by node id, valued by that node's
    child ids) contain a cycle. `related` edges are exempt (S2.5) and must not be passed here.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(child_edges, WHITE)

    def visit(node: str, path: list[str]) -> None:
        color[node] = GRAY
        for child in child_edges.get(node, ()):
            if child not in color:
                continue
            if color[child] == GRAY:
                cycle = " -> ".join((*path, child))
                raise ValueError(f"Cycle detected in parent/child edges: {cycle}")
            if color[child] == WHITE:
                visit(child, [*path, child])
        color[node] = BLACK

    for node in child_edges:
        if color[node] == WHITE:
            visit(node, [node])
