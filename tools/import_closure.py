"""Static (AST-only) import closure of entry scripts in the original repository.

Reads source text only; never imports or executes the analysed code.

Usage:
    python tools/import_closure.py <original_repo_root> <entry.py> [<entry.py> ...]

Prints one repo-relative path per line (the transitive closure of
`cardio_organoid.*` modules and sibling `scripts/*.py` modules reached via the
original scripts' `sys.path.insert(scripts_dir)` convention), plus a list of
third-party top-level imports.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def _module_file(repo: Path, dotted: str) -> Path | None:
    parts = dotted.split(".")
    if parts[0] == "cardio_organoid":
        base = repo / "src" / Path(*parts)
        for cand in (base.with_suffix(".py"), base / "__init__.py"):
            if cand.is_file():
                return cand
        return None
    cand = repo / "scripts" / f"{parts[0]}.py"
    return cand if cand.is_file() else None


def _imports(path: Path) -> list[tuple[str, list[str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((a.name, []) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.module, [a.name for a in node.names]))
    return out


def closure(repo: Path, entries: list[Path]) -> tuple[list[Path], set[str]]:
    seen: set[Path] = set()
    third_party: set[str] = set()
    stack = [p.resolve() for p in entries]
    stdlib = set(sys.stdlib_module_names)
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        # package __init__ files along the path are imported implicitly
        if f.is_relative_to(repo / "src"):
            d = f.parent
            while d != repo / "src":
                init = d / "__init__.py"
                if init.is_file() and init not in seen:
                    stack.append(init)
                d = d.parent
        for mod, names in _imports(f):
            target = _module_file(repo, mod)
            if target is not None:
                stack.append(target)
                for n in names:  # `from pkg import submodule`
                    sub = _module_file(repo, f"{mod}.{n}")
                    if sub is not None:
                        stack.append(sub)
            else:
                top = mod.split(".")[0]
                if top not in stdlib and top != "cardio_organoid" and top != "__future__":
                    third_party.add(top)
    return sorted(seen), third_party


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    files, third = closure(repo, [Path(a) for a in sys.argv[2:]])
    for f in files:
        print(f.relative_to(repo))
    print("# third-party:", " ".join(sorted(third)))


if __name__ == "__main__":
    main()
