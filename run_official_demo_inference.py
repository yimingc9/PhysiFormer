#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from official_demo_inference.launcher import main


if __name__ == "__main__":
    raise SystemExit(main())
