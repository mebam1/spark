#!/usr/bin/env python3
"""Articulation-only SPARK URDF refinement entrypoint."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from URDFoptimizer.spark_articulation.cli import main


if __name__ == "__main__":
    main()
