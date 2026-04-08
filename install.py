#!/usr/bin/env python3
"""
install.py — Bootstrap installer for obsidian-llm-wiki (olw).

Usage:
    python install.py           # auto-detect uv (preferred) or pip
    python install.py --pip     # force pip install -e .
    python install.py --uv      # force uv tool install .
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Color helpers (stdlib only, no deps) ──────────────────────────────────────


def _windows_ansi_enabled() -> bool:
    """Enable VT processing on Windows 10+ and return True only if SetConsoleMode succeeds."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        stdout = kernel32.GetStdHandle(-11)
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004, combined with default 0x0003
        return kernel32.SetConsoleMode(stdout, 7) != 0
    except Exception:
        return False


USE_COLOR = sys.stdout.isatty() and (os.name != "nt" or _windows_ansi_enabled())


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def red(t: str) -> str:
    return _c(t, "31")


def bold(t: str) -> str:
    return _c(t, "1")


def dim(t: str) -> str:
    return _c(t, "2")


def info(msg: str) -> None:
    print(f"  {msg}")


def ok(msg: str) -> None:
    print(f"  {green('✓')} {msg}")


def warn(msg: str) -> None:
    print(f"  {yellow('!')} {msg}")


def err(msg: str) -> None:
    print(f"  {red('✗')} {msg}", file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    err(msg)
    sys.exit(code)


def rule(char: str = "─", width: int = 50) -> None:
    print(dim(char * width))


# ── Preflight checks ──────────────────────────────────────────────────────────


def check_python() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        die(
            f"Python 3.11+ required. You have {major}.{minor}.\n"
            "  Download from https://python.org/downloads/"
        )
    ok(f"Python {major}.{minor}")


def check_ollama() -> None:
    """Non-fatal: warns if ollama binary not found."""
    if shutil.which("ollama"):
        ok("ollama found")
    else:
        warn("ollama not found — install from https://ollama.com/download")
        warn("You can install it later; olw will work once ollama is running.")


def detect_repo_root() -> Path:
    here = Path(__file__).parent.resolve()
    if not (here / "pyproject.toml").exists():
        die(
            "Run install.py from the repo root directory (where pyproject.toml lives).\n"
            f"  Current directory: {here}"
        )
    return here


# ── Installation ──────────────────────────────────────────────────────────────


def detect_installer(force_uv: bool, force_pip: bool) -> str:
    if force_pip:
        return "pip"
    if force_uv:
        if not shutil.which("uv"):
            die(
                "uv not found. Install from https://docs.astral.sh/uv/ "
                "or use: python install.py --pip"
            )
        return "uv"
    return "uv" if shutil.which("uv") else "pip"


def install_with_uv(repo_root: Path) -> None:
    info("Running: uv tool install . --force")
    result = subprocess.run(["uv", "tool", "install", ".", "--force"], cwd=repo_root)
    if result.returncode != 0:
        die("uv install failed. Try: python install.py --pip")


def install_with_pip(repo_root: Path) -> None:
    info("Running: pip install -e .")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo_root)
    if result.returncode != 0:
        die("pip install failed. See errors above.")


# ── Verification ──────────────────────────────────────────────────────────────


def verify_install() -> bool:
    """Check that `olw` is on PATH and responds to --version."""
    olw = shutil.which("olw")
    if not olw:
        return False
    result = subprocess.run(["olw", "--version"], capture_output=True, text=True)
    return result.returncode == 0


def fix_windows_path_hint(installer: str) -> None:
    """On Windows, print the exact directory to add to PATH."""
    if os.name != "nt":
        return
    if installer == "uv":
        tools_dir = "~\\.local\\bin"
        try:
            r = subprocess.run(["uv", "tool", "dir"], capture_output=True, text=True)
            if r.returncode == 0:
                tools_dir = r.stdout.strip()
        except Exception:
            pass
        warn(f"Add to PATH: {tools_dir}")
        warn("In PowerShell:  $env:PATH += ';' + (uv tool dir)")
        warn("Permanently:    [Environment]::SetEnvironmentVariable('PATH', $env:PATH, 'User')")
    else:
        scripts = Path(sys.prefix) / "Scripts"
        warn(f"Add to PATH: {scripts}")
        warn(f"Or run directly: {scripts / 'olw.exe'}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Install obsidian-llm-wiki (olw)")
    parser.add_argument("--pip", action="store_true", help="Force pip install")
    parser.add_argument("--uv", action="store_true", help="Force uv install")
    args = parser.parse_args()

    print()
    print(bold("obsidian-llm-wiki") + "  installer")
    rule()
    print()

    check_python()
    repo_root = detect_repo_root()
    check_ollama()

    installer = detect_installer(force_uv=args.uv, force_pip=args.pip)
    info(f"Package manager: {bold(installer)}")
    print()

    if installer == "uv":
        install_with_uv(repo_root)
    else:
        install_with_pip(repo_root)

    print()

    if verify_install():
        ok("olw is on PATH and working")
        print()
        rule("━")
        print()
        ok(bold("Installation complete!"))
        print()
        info("Run the interactive setup wizard:")
        print()
        print(f"    {bold('olw setup')}")
        print()
        info("This configures your Ollama URL, models, and default vault.")
        print()
    else:
        warn("olw was installed but not found on PATH.")
        fix_windows_path_hint(installer)
        print()
        info("After updating PATH, run:")
        print(f"    {bold('olw setup')}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
