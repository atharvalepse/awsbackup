"""Cross-platform dev runner. Same commands work in PowerShell, cmd, bash, zsh.

Usage:
    python scripts/dev.py setup     # install editable + dev extras, write .env
    python scripts/dev.py test      # run pytest
    python scripts/dev.py metrics   # serve /metrics on :9090 via uvicorn
    python scripts/dev.py clean     # remove build/cache artifacts

Stdlib-only; no external dependencies. Always invokes the *current*
interpreter (`sys.executable`), so it does the right thing whether the venv
is activated or you call it as `.venv\\Scripts\\python scripts\\dev.py …`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> int:
    print("$", " ".join(args), flush=True)
    return subprocess.call(args, cwd=REPO)


def cmd_setup(_: argparse.Namespace) -> int:
    rc = _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    if rc:
        return rc
    rc = _run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
    if rc:
        return rc
    env, template = REPO / ".env", REPO / ".env.example"
    if not env.exists() and template.exists():
        shutil.copyfile(template, env)
        print(f"wrote {env} from {template.name}")
    return 0


def cmd_test(ns: argparse.Namespace) -> int:
    extra = list(ns.pytest_args)
    # argparse.REMAINDER preserves a leading `--`; pytest treats it as
    # end-of-options and starts interpreting flags like `-q` as file paths.
    if extra and extra[0] == "--":
        extra = extra[1:]
    return _run([sys.executable, "-m", "pytest", *extra])


def cmd_metrics(ns: argparse.Namespace) -> int:
    return _run([
        sys.executable, "-m", "uvicorn",
        "orchestration.metrics_endpoint:app",
        "--host", ns.host, "--port", str(ns.port),
    ])


_CLEAN_DIRS = (
    "build", "dist", ".pytest_cache", ".mypy_cache",
    "gml_orchestration.egg-info",
)
_CLEAN_GLOBS = ("**/__pycache__", "**/*.pyc", "**/*.pyo")


def cmd_clean(_: argparse.Namespace) -> int:
    removed = 0
    for d in _CLEAN_DIRS:
        p = REPO / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"removed {p}")
            removed += 1
    for pattern in _CLEAN_GLOBS:
        for p in REPO.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            removed += 1
    print(f"cleaned {removed} item(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="dev", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="install editable + dev extras, seed .env").set_defaults(func=cmd_setup)

    p_test = sub.add_parser("test", help="run pytest (extra args forwarded)")
    p_test.add_argument("pytest_args", nargs=argparse.REMAINDER)
    p_test.set_defaults(func=cmd_test)

    p_metrics = sub.add_parser("metrics", help="serve /metrics via uvicorn")
    p_metrics.add_argument("--host", default="127.0.0.1")
    p_metrics.add_argument("--port", type=int, default=9090)
    p_metrics.set_defaults(func=cmd_metrics)

    sub.add_parser("clean", help="remove build/cache artifacts").set_defaults(func=cmd_clean)

    ns = parser.parse_args()
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
