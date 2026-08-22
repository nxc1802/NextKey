#!/usr/bin/env python3
"""Runner script for NextKey Streamlit Frontend Demo."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
APP_PATH = SRC / "FE" / "app.py"


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}"

    port = 8501
    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])

    print(f"🚀 Launching NextKey Streamlit UI at http://localhost:{port}")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 NextKey Streamlit UI stopped.")


if __name__ == "__main__":
    main()
