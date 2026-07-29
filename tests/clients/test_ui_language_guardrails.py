from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UI_PATHS = (
    ROOT / "src" / "archivetrust" / "clients" / "desktop",
    ROOT / "src" / "archivetrust" / "clients" / "desktop_v2",
)
FORBIDDEN_VISIBLE_COPY = ("ROADMAP.md", "Part 8", "Part 9", "§5.", "§9", "§10")


def test_ui_visible_strings_do_not_cite_internal_specs() -> None:
    offenders: list[str] = []
    for root in UI_PATHS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            docstring_nodes = set(_docstring_nodes(tree))
            for node in ast.walk(tree):
                if node in docstring_nodes:
                    continue
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for snippet in FORBIDDEN_VISIBLE_COPY:
                        if snippet in node.value:
                            rel = path.relative_to(ROOT)
                            offenders.append(f"{rel}:{node.lineno}: {snippet}")

    assert offenders == []


def _docstring_nodes(tree: ast.AST) -> list[ast.Constant]:
    nodes: list[ast.Constant] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    nodes.append(value)
    return nodes
