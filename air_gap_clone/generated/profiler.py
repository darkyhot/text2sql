#!/usr/bin/env python3
"""Точка входа программы 1 (закрытый контур). См. agc_profiler/cli.py.

Запуск:  python profiler.py --tables "public.tasks,public.clients" --out profile.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agc_profiler.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
