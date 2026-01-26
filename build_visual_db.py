"""
Convenience script to build the trusted screenshot database.
Run from the project root: python build_visual_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.visual_detection.build_trusted_db import build_trusted_database

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PhishGuard AI — Trusted Screenshot Database Builder")
    print("=" * 60)
    print("This will capture screenshots of trusted websites.")
    print("Requires: Google Chrome installed on your system.\n")
    build_trusted_database()
