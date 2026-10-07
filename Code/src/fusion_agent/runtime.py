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
    """Estimate extraction, association, verification and recovery requests."""
    folders = IngestionAgent(settings.paths.data_dir).dates()
    text_count = sum(sum(1 for path in (folders[day] / name).glob("*.txt") if not path.name.startswith("~$"))
                     for day in set(targets) for name in ("开源军情", "网上舆论"))
    news_count = sum(sum(1 for path in (folders[day] / "开源军情").glob("*.txt") if not path.name.startswith("~$"))
                     for day in set(targets))
    if settings.alignment.mode == "one_to_one":
        reviews = news_count if settings.alignment.llm_review else 0
        verifications = news_count
    else:
        # Legacy all-combinations mode has no bounded final model review count.
        reviews, verifications = 0, 0
    extra = 0
    if (settings.workflow.enabled and settings.alignment.mode == "one_to_one"
            and settings.alignment.llm_review and settings.extraction.mode != "rules"):
        rounds = settings.workflow.max_recovery_rounds
        # Allow source completeness and name ambiguity checks, new link/fact reviews after each
        # recovery round, and one exception search per missing source type.
        extra = text_count * 2 + rounds * (reviews + verifications + news_count * 2)
    return max(1, (text_count + reviews + verifications + extra) * (settings.llm.retries + 1))
