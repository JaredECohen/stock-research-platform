"""Prove a typing sweep changed nothing but annotations.

RP-005 rewrote ~2,500 annotations (`Dict[str, X]` -> `dict[str, X]`,
`Optional[X]` -> `X | None`), dropped the `typing` imports that became unused
and re-sorted import blocks. Every one of those edits is invisible at runtime
under `from __future__ import annotations`, but "invisible" is a claim, and a
2,000-line diff is not something a reviewer can verify by eye. This script
turns the claim into a check: for each file that differs from the base
revision it parses both versions, erases everything the sweep was allowed to
touch, and asserts the two ASTs are identical. Anything the sweep was *not*
allowed to touch survives the erasure and shows up as a difference.

What is erased (all of it documented because it is exactly the trust
boundary of the check):

* annotations on arguments, returns and assignments (`x: T = v` becomes
  `x = v`; a bare `x: T` keeps its position as a placeholder so dataclass /
  pydantic field order is still compared);
* `from __future__ import annotations`, `import typing`, `from typing ...`
  and `from collections.abc ...` statements (the UP035 move lands there);
* the order of import statements within one contiguous block, and whether
  two `from x import ...` lines were merged into one -- ruff's isort does
  both; imports are compared as a set of (module, name, alias, level);
* `typing` aliases used in runtime expressions (`cast(Dict[str, Any], v)`),
  which ruff rewrites in the same way it rewrites annotations: `Dict` ->
  `dict`, `Optional[X]` -> `X | None`, `Union[A, B]` -> `A | B`, and so on.

Everything else -- every statement, expression, default value, decorator,
string literal -- must match `ast.dump` for `ast.dump`.

Usage, from `backend/`:

    python scripts/ast_equivalence_check.py --base 724b6bb            # all changed files
    python scripts/ast_equivalence_check.py --base HEAD~1 app/x.py     # a subset
    python scripts/ast_equivalence_check.py --dump-schemas out.json    # pydantic snapshot
    python scripts/ast_equivalence_check.py --compare-schemas out.json

`--dump-schemas` / `--compare-schemas` cover the one place the annotations
*are* evaluated at runtime: pydantic builds `model_json_schema()` from them,
so `StockMemoOut` and `DCFResult` (the two persisted contracts) are
snapshotted before the sweep and compared after it. Exit status is non-zero
on any difference.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
SCHEMA_MODELS = ("StockMemoOut", "DCFResult")

# typing alias -> builtin, exactly the UP006 table for names this codebase used.
_BUILTIN_ALIASES = {
    "Dict": "dict",
    "List": "list",
    "Tuple": "tuple",
    "Set": "set",
    "FrozenSet": "frozenset",
    "Type": "type",
    "Deque": "deque",
    "DefaultDict": "defaultdict",
}
_ERASED_IMPORT_MODULES = {"__future__", "typing", "collections.abc"}


class _Normalize(ast.NodeTransformer):
    """Erase annotations and canonicalise typing aliases in runtime code."""

    # -- annotations --------------------------------------------------------
    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return self.generic_visit(node)

    def _strip_function(self, node):
        node.returns = None
        return self.generic_visit(node)

    visit_FunctionDef = _strip_function
    visit_AsyncFunctionDef = _strip_function

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        if node.value is not None:
            return ast.copy_location(
                ast.Assign(targets=[node.target], value=node.value, type_comment=None), node
            )
        # Bare `name: T` -- keep the name in place so field order is checked.
        return ast.copy_location(
            ast.Expr(value=ast.Constant(value=f"<bare annotation {ast.dump(node.target)}>")), node
        )

    # -- typing aliases in runtime expressions -------------------------------
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in _BUILTIN_ALIASES:
            return ast.copy_location(ast.Name(id=_BUILTIN_ALIASES[node.id], ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        # `typing.Dict` -> `dict`
        if isinstance(node.value, ast.Name) and node.value.id == "typing" and node.attr in _BUILTIN_ALIASES:
            return ast.copy_location(ast.Name(id=_BUILTIN_ALIASES[node.attr], ctx=node.ctx), node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        head = node.value
        name = None
        if isinstance(head, ast.Name):
            name = head.id
        elif isinstance(head, ast.Attribute) and isinstance(head.value, ast.Name) and head.value.id == "typing":
            name = head.attr
        if name == "Optional":
            return ast.copy_location(
                ast.BinOp(left=node.slice, op=ast.BitOr(), right=ast.Constant(value=None)), node
            )
        if name == "Union":
            elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            expr = elts[0]
            for elt in elts[1:]:
                expr = ast.BinOp(left=expr, op=ast.BitOr(), right=elt)
            return ast.copy_location(expr, node)
        return node


def _import_key(node: ast.AST):
    if isinstance(node, ast.Import):
        return [("import", alias.name, alias.asname, 0) for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [(node.module or "", alias.name, alias.asname, node.level) for alias in node.names]
    return None


def _is_erased_import(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return (node.module or "") in _ERASED_IMPORT_MODULES
    if isinstance(node, ast.Import):
        return all(alias.name in _ERASED_IMPORT_MODULES for alias in node.names)
    return False


def _fold_import_blocks(tree: ast.AST) -> None:
    """Replace each contiguous run of imports with one order-insensitive node."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody", "handlers"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list) or not stmts or not isinstance(stmts[0], ast.stmt):
                continue
            out: list[ast.stmt] = []
            block: list[tuple] = []
            for stmt in stmts:
                keys = _import_key(stmt)
                if keys is None:
                    _flush_import_block(block, out)
                    out.append(stmt)
                elif not _is_erased_import(stmt):
                    block.extend(keys)
            _flush_import_block(block, out)
            setattr(node, field, out)


def _flush_import_block(block: list[tuple], out: list[ast.stmt]) -> None:
    if block:
        out.append(ast.Expr(value=ast.Constant(value="<imports> " + repr(sorted(block)))))
        block.clear()


def normalized_dump(source: str, filename: str) -> str:
    tree = ast.parse(source, filename)
    tree = _Normalize().visit(tree)
    _fold_import_blocks(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, indent=1)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=BACKEND_DIR).stdout


def changed_python_files(base: str) -> list[str]:
    toplevel = Path(_git("rev-parse", "--show-toplevel").strip())
    # Absolute pathspec: git resolves it regardless of the cwd it runs from.
    names = _git("diff", "--name-only", "--diff-filter=AMR", base, "--", str(BACKEND_DIR / "app")).split()
    return [os.path.relpath(toplevel / n, BACKEND_DIR) for n in names if n.endswith(".py")]


def check_files(base: str, files: list[str], verbose: bool) -> int:
    toplevel = Path(_git("rev-parse", "--show-toplevel").strip())
    failures = 0
    identical = 0
    for rel in files:
        path = (BACKEND_DIR / rel).resolve()
        repo_rel = path.relative_to(toplevel).as_posix()
        try:
            before_src = _git("show", f"{base}:{repo_rel}")
        except subprocess.CalledProcessError:
            print(f"NEW      {rel} (not in {base}; nothing to compare)")
            continue
        after_src = path.read_text()
        before = normalized_dump(before_src, f"{base}:{repo_rel}")
        after = normalized_dump(after_src, str(path))
        if before == after:
            identical += 1
            if verbose:
                print(f"ok       {rel}")
            continue
        failures += 1
        print(f"DIFFERS  {rel}")
        diff = difflib.unified_diff(before.splitlines(), after.splitlines(), f"{base}:{repo_rel}", rel, lineterm="", n=2)
        for line in list(diff)[:80]:
            print("    " + line)
    print(f"\n{identical} file(s) equivalent modulo annotations/typing imports, {failures} differ, "
          f"{len(files)} checked against {base}.")
    return 1 if failures else 0


def _schema_snapshot() -> dict[str, str]:
    sys.path.insert(0, str(BACKEND_DIR))
    from app import schemas  # noqa: PLC0415 - deliberate late import, only for the schema mode

    out = {}
    for name in SCHEMA_MODELS:
        text = json.dumps(getattr(schemas, name).model_json_schema(), sort_keys=True)
        out[name] = hashlib.sha256(text.encode()).hexdigest()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="HEAD", help="git revision holding the 'before' sources (default HEAD)")
    parser.add_argument("files", nargs="*", help="files under backend/ to check (default: every changed .py)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the files that pass too")
    parser.add_argument("--dump-schemas", metavar="PATH", help="write the pydantic schema hashes to PATH and exit")
    parser.add_argument("--compare-schemas", metavar="PATH", help="compare current pydantic schema hashes with PATH")
    args = parser.parse_args(argv)

    if args.dump_schemas:
        snap = _schema_snapshot()
        Path(args.dump_schemas).write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
        print(json.dumps(snap, indent=1, sort_keys=True))
        return 0
    if args.compare_schemas:
        expected = json.loads(Path(args.compare_schemas).read_text())
        actual = _schema_snapshot()
        status = 0
        for name in SCHEMA_MODELS:
            same = expected.get(name) == actual[name]
            print(f"{'ok      ' if same else 'DIFFERS '} {name}.model_json_schema() sha256={actual[name][:16]}...")
            status |= 0 if same else 1
        return status

    files = args.files or changed_python_files(args.base)
    if not files:
        print(f"no python files under backend/app differ from {args.base}")
        return 0
    return check_files(args.base, files, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
