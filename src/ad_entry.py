from __future__ import annotations

import subprocess
import sys

from cli import main


def _ensure_qdrant() -> None:
    """Start Qdrant Docker container if not already running."""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return

    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=False,
    )
    names = result.stdout.strip().splitlines()
    container_exists = "auraderma-qdrant" in names

    if not container_exists:
        subprocess.run(
            ["docker", "compose", "-f", "docker-compose.yml", "up", "-d", "qdrant"],
            check=False,
        )
    else:
        running = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=False,
        )
        if "auraderma-qdrant" not in running.stdout.strip().splitlines():
            subprocess.run(["docker", "start", "auraderma-qdrant"], check=False)


def ad_main() -> None:
    """Entry point for the 'ad' command — launch chat (default) or forward args."""
    if len(sys.argv) <= 1:
        # No args: run chat
        _ensure_qdrant()
        sys.argv = [sys.argv[0], "chat"]
    main()
