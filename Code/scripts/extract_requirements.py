"""将赛题 PDF 的可检索文本导出到 Code/outputs，方便核对输入口径。"""
import argparse
from pathlib import Path
import sys

import pymupdf

from fusion_agent.utils import write_text


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=root/"Code"/"outputs"/"competition_requirements.txt")
    args = parser.parse_args()
    if args.pdf is None:
        paths = list((root/"Docs").glob("*.pdf"))
        if len(paths) != 1:
            parser.error("请使用 --pdf 指定赛题 PDF")
        args.pdf = paths[0]
    with pymupdf.open(args.pdf) as doc:
        text = "\n\n".join(f"PAGE {i+1}\n{p.get_text()}" for i,p in enumerate(doc))
    write_text(args.out, text)
    print(f"PDF text exported to: {args.out}")


if __name__ == "__main__":
    main()

