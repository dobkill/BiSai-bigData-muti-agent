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
    parser.add_argument("--workers", type=int, default=4, help="并发推理请求数；须配合服务端OLLAMA_NUM_PARALLEL")
    parser.add_argument("--num-ctx", type=int, default=12288)
    parser.add_argument("--chat-url", help="可选的llama-server文本服务；嵌入仍使用Ollama")
    parser.add_argument("--cache-run", help="使用独立模型缓存目录；新名称会重新推理，重复名称可断点续跑")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
