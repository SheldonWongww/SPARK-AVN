#!/usr/bin/env python3
"""Read-only lexical inventory of implementation surfaces relevant to a TTA port."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md", ".sh"}
SKIP_PARTS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules"}
SECRET_LINE = re.compile(
    r"(?:password|passwd|api[_-]?key|access[_-]?token|private[_-]?key|cookie)",
    re.I,
)
CATEGORIES = {
    "entrypoint_config": re.compile(r"\b(main|train|eval|runner|config|registry|build_adapter)\b", re.I),
    "action": re.compile(r"\b(action|argmax|sample|categorical|stop|mask|candidate|logits?)\b", re.I),
    "feedback": re.compile(r"\b(feedback|success|failure|reward|oracle|query|human)\b", re.I),
    "update": re.compile(r"\b(backward|optimizer|step|zero_grad|loss|learning_rate|\blr\b|update_interval)\b", re.I),
    "parameters_freeze": re.compile(r"\b(requires_grad|trainable|freeze|frozen|parameter|param_scope|named_parameters)\b", re.I),
    "state_reset": re.compile(r"\b(reset|episodic|continual|hidden_state|recurrent|buffer|state_dict|load_state_dict)\b", re.I),
    "replay_memory": re.compile(r"\b(replay|memory|reservoir|queue|cache|detach|clone|snapshot)\b", re.I),
    "randomness": re.compile(r"\b(seed|rng|random|generator|deterministic|dropout|shuffle)\b", re.I),
    "provenance_output": re.compile(r"\b(manifest|sha256|checkpoint|dataset|commit|metric|diagnostic|result)\b", re.I),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan source text for TTA port surfaces; never modifies the target tree."
    )
    parser.add_argument("path", type=Path, help="File or directory to scan")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--max-hits-per-category", type=int, default=40)
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    return parser.parse_args()


def iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            if not any(part in SKIP_PARTS for part in path.parts):
                yield path


def python_symbols(text: str) -> list[dict[str, object]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    symbols: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            symbols.append({"kind": kind, "name": node.name, "line": node.lineno})
    return sorted(symbols, key=lambda item: (int(item["line"]), str(item["name"])))


def display_path(path: Path, root: Path) -> str:
    if root.is_file():
        return path.name
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def scan(root: Path, max_hits: int, max_bytes: int) -> tuple[dict[str, object], bool]:
    hits: dict[str, list[dict[str, object]]] = defaultdict(list)
    symbols: dict[str, list[dict[str, object]]] = {}
    suffixes: Counter[str] = Counter()
    skipped: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    scanned = 0
    sensitive_lines_omitted = 0

    for path in iter_files(root):
        rel = display_path(path, root)
        try:
            size = path.stat().st_size
            if size > max_bytes:
                skipped.append({"path": rel, "reason": "size_limit", "bytes": size})
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append({"path": rel, "error": str(exc)})
            continue
        scanned += 1
        suffixes[path.suffix.lower() or "<none>"] += 1
        if path.suffix.lower() == ".py":
            found = python_symbols(text)
            if found:
                symbols[rel] = found
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if SECRET_LINE.search(line):
                sensitive_lines_omitted += 1
                continue
            for category, pattern in CATEGORIES.items():
                if len(hits[category]) < max_hits and pattern.search(line):
                    hits[category].append(
                        {"path": rel, "line": line_number, "text": line[:240]}
                    )

    report: dict[str, object] = {
        "root": str(root.resolve()),
        "read_only": True,
        "files_scanned": scanned,
        "files_by_suffix": dict(sorted(suffixes.items())),
        "categories": {name: hits.get(name, []) for name in CATEGORIES},
        "python_symbols": symbols,
        "skipped": skipped,
        "errors": errors,
        "sensitive_lines_omitted": sensitive_lines_omitted,
        "notice": "Lexical matches are navigation aids; verify semantics by tracing executed control flow.",
    }
    return report, bool(errors)


def render_text(report: dict[str, object]) -> str:
    lines = [
        f"root: {report['root']}",
        "read_only: true",
        f"files_scanned: {report['files_scanned']}",
        "files_by_suffix: " + json.dumps(report["files_by_suffix"], sort_keys=True),
    ]
    categories = report["categories"]
    assert isinstance(categories, dict)
    for name in CATEGORIES:
        category_hits = categories[name]
        lines.append(f"\n[{name}] hits={len(category_hits)}")
        for hit in category_hits:
            lines.append(f"{hit['path']}:{hit['line']}: {hit['text']}")
    symbols = report["python_symbols"]
    assert isinstance(symbols, dict)
    lines.append(f"\n[python_symbols] files={len(symbols)}")
    for path, items in symbols.items():
        compact = ", ".join(f"{item['kind']} {item['name']}:{item['line']}" for item in items)
        lines.append(f"{path}: {compact}")
    skipped = report["skipped"]
    errors = report["errors"]
    lines.append(
        "\nskipped={} sensitive_lines_omitted={} errors={}".format(
            len(skipped), report["sensitive_lines_omitted"], len(errors)
        )
    )
    for item in errors:
        lines.append(f"ERROR {item['path']}: {item['error']}")
    lines.append(str(report["notice"]))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.max_hits_per_category < 1 or args.max_file_bytes < 1:
        print("limits must be positive", file=sys.stderr)
        return 2
    if not args.path.exists():
        print(f"path does not exist: {args.path}", file=sys.stderr)
        return 2
    report, had_errors = scan(args.path, args.max_hits_per_category, args.max_file_bytes)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
