"""Run from repository root: python Code/match_day.py --input Datas/数据/2025.5.1 --output Datas/匹配数据."""
import argparse
import sys
from pathlib import Path

from event_matcher.pipeline import run


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description="本地Ollama三源事件配对，保留原文、异常和未匹配材料")
    parser.add_argument("--input", type=Path, default=Path("Datas/数据/2025.5.1"))
    parser.add_argument("--output", type=Path, default=Path("Datas/匹配数据"))
    parser.add_argument("--model", default="qwen3.5:9b-q8_0")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--stage", choices=["extract", "all"], default="all")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
