"""Shared command-line backend overrides and full-run request budgeting."""
from .config import LLMSettings
from .ingestion import IngestionAgent


def add_llm_arguments(parser):
    parser.add_argument("--provider", choices=["ollama", "llama_cpp", "api"], help="本次模型后端，覆盖配置")
    parser.add_argument("--model", help="本次后端模型名/llama-server alias")
    parser.add_argument("--base-url", help="本次后端地址；llama_cpp/API 应包含 /v1")
    parser.add_argument("--concurrency", type=int, help="固定并发数或动态并发上限（1–32）")
    adaptive = parser.add_mutually_exclusive_group()
    adaptive.add_argument("--adaptive-concurrency", action="store_true", help="本机 llama_cpp 从低并发逐级测试显存和吞吐")
    adaptive.add_argument("--fixed-concurrency", action="store_true", help="关闭动态调度，使用固定并发数")


def apply_llm_arguments(settings, args):
    llm = settings.llm
    if args.provider:
        llm.provider = args.provider
        # Switching back to an existing backend should not require editing adaptive TOML.
        if llm.provider != "llama_cpp":
            llm.adaptive.enabled = False
    if args.model:
        llm.endpoint.model = args.model
    if args.base_url:
        llm.endpoint.base_url = args.base_url
    if args.concurrency is not None:
        llm.concurrency = args.concurrency
    if args.adaptive_concurrency:
        llm.adaptive.enabled = True
    if args.fixed_concurrency:
        llm.adaptive.enabled = False
    settings.llm = LLMSettings.model_validate(llm.model_dump())


def estimate_request_budget(settings, targets) -> int:
    """Cover each actual text file and per-day reviews, including allowed retries."""
    folders = IngestionAgent(settings.paths.data_dir).dates()
    text_count = sum(sum(1 for path in (folders[day] / name).glob("*.txt") if not path.name.startswith("~$"))
                     for day in set(targets) for name in ("开源军情", "网上舆论"))
    reviews = len(set(targets)) * settings.alignment.max_llm_reviews if settings.alignment.llm_review else 0
    return max(1, (text_count + reviews) * (settings.llm.retries + 1))
