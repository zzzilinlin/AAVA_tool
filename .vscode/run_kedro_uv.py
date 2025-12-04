"""Helper to run `uv run kedro run` so VS Code can launch it via the Python debugger.

This script shells out to the `uv` command. The process will inherit the
integrated terminal so you can see output and interact if needed.
"""
import subprocess
import sys

cmd = ["uv", "run", "kedro", "run"]

try:
    rc = subprocess.call(cmd)
except FileNotFoundError:
    print("Command 'uv' not found in PATH. If you prefer, use the 'Run Kedro (python -m kedro run)' launch configuration.")
    rc = 1

sys.exit(rc)
