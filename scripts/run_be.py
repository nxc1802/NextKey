#!/usr/bin/env python3
"""Runner script for NextKey FastAPI Backend."""

import argparse
import os
import sys
from pathlib import Path

# Add src to python path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run NextKey FastAPI Backend")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    args = parser.parse_args()

    os.environ["PYTHONPATH"] = str(SRC)
    print(f"🚀 Starting NextKey FastAPI Backend at http://{args.host}:{args.port}")
    print(f"📚 Swagger Documentation: http://localhost:{args.port}/docs")
    uvicorn.run("BE.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
