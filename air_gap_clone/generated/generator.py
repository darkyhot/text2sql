#!/usr/bin/env python3
"""Точка входа программы 2 (открытый контур). См. agc_generator/cli.py.

Запуск:  python generator.py --profile profile.json --scale 0.001 --format csv --out out/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agc_generator.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
