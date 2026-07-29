"""Contract test for ROADMAP.md S8 Milestone 1 acceptance criterion: "No import of any
providers/* module from domain/evidence, domain/ontology, domain/graph, or domain/document."

There is no `providers/` package yet (Milestone 3), so this test also guards against it being
introduced later with an accidental inward dependency -- it greps the actual domain source, not a
snapshot, so it stays true as the codebase grows.
"""

from __future__ import annotations

import ast
from pathlib import Path

DOMAIN_ROOT = Path(__file__).resolve().parents[2] / "src" / "archivetrust" / "domain"


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_domain_layer_never_imports_providers():
    python_files = list(DOMAIN_ROOT.rglob("*.py"))
    assert python_files, "expected domain source files to exist"

    offenders = []
    for path in python_files:
        for module_name in _imported_module_names(path):
            if module_name.startswith("archivetrust.providers") or module_name == "providers":
                offenders.append((path, module_name))

    assert not offenders, f"domain layer imports providers/*: {offenders}"


def test_domain_layer_has_zero_third_party_runtime_dependencies_beyond_pydantic():
    # ROADMAP.md S11 Technical Decisions: "Domain layer has zero third-party runtime dependencies
    # beyond schema/validation library." pydantic (and its own transitive deps) is the only
    # allowed third-party import; everything else must be stdlib or archivetrust-internal.
    allowed_prefixes = ("pydantic", "archivetrust", "__future__")
    stdlib_names = {
        "enum",
        "hashlib",
        "uuid",
        "typing",
        "collections",
        "ast",
        "pathlib",
        "difflib",
        "re",
        "unicodedata",
        "datetime",
        "dataclasses",
        "math",
    }

    offenders = []
    for path in DOMAIN_ROOT.rglob("*.py"):
        for module_name in _imported_module_names(path):
            top_level = module_name.split(".")[0]
            if module_name.startswith(allowed_prefixes) or top_level in stdlib_names:
                continue
            offenders.append((path, module_name))

    assert not offenders, f"domain layer imports non-stdlib, non-pydantic dependency: {offenders}"
