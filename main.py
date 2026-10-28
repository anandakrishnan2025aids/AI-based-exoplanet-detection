#!/usr/bin/env python
"""
ExoMiner Main Entry Point
Forwarding script to run.py.
"""
import sys
from pathlib import Path

# Add workspace directory to path and execute run.py
workspace_dir = Path(__file__).resolve().parent
if str(workspace_dir) not in sys.path:
    sys.path.insert(0, str(workspace_dir))

from run import main

if __name__ == "__main__":
    main()
