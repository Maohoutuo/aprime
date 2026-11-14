# -*- coding: utf-8 -*-
"""命令行入口：python -m agentsense <command>。"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"

COMMANDS = {
    "smoke": "smoke.py",
    "calibrate": "calibrate.py",
    "run-table4": "run_table4.py",
    "quick-scale": "quick_scale.py",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentsense", description="AgentSense 复现工程")
    parser.add_argument("command", choices=sorted(COMMANDS), help="要运行的实验脚本")
    args, rest = parser.parse_known_args(argv)
    script = EXPERIMENTS / COMMANDS[args.command]
    sys.argv = [str(script)] + rest
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
