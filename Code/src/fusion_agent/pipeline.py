from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import time

from . import __version__
from .agents.alignment import AlignmentAgent
from .agents.extraction import ExtractionAgent
from .agents.event_units import EventUnitAgent
from .agents.report import ReportAgent
from .agents.temporal import TemporalAgent
from .agents.verification import VerificationAgent
from .config import Settings
from .ingestion import IngestionAgent
from .knowledge import build_graph
from .evidence_store import EvidenceStore
from .llm import LLMClient
from .models import DayResult
from .review_export import ReviewExporter
from .utils import stable_id, write_json, write_text

logger = logging.getLogger(__name__)


class AgentPipeline:
    """共享结构化状态的 Agent 工作流；模型不能跳过核验直接写入事实。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ingestion = IngestionAgent(settings.paths.data_dir)
        self.llm = LLMClient(settings.llm, settings.paths.cache_dir) if settings.llm.enabled and settings.extraction.mode != "rules" else None
        if self.llm:
            self.llm.validate()
        self.extraction = ExtractionAgent(settings, self.llm)
        self.event_units = EventUnitAgent()
        self.alignment = AlignmentAgent(settings.alignment, self.llm)
        self.store = EvidenceStore(settings.paths.output_dir / "evidence.sqlite")
        self.verification = VerificationAgent(settings.verification, settings.alignment)
        self.temporal = TemporalAgent(settings.report.lookback_days)
        self.report = ReportAgent()

    def close(self):
        self.store.close()
        if self.llm:
            self.llm.close()

    def cache_key(self, day: str) -> str:
        values = self.settings.model_dump(mode="json", exclude={"paths", "review_export"})
        # 排除密钥；模型、服务地址、提示词、算法参数、数据内容都参与版本计算。
        values["llm"]["api"].pop("api_key", None)
        source_hash = hashlib.sha256()
        for path in sorted(Path(__file__).parent.rglob("*.py")):
            source_hash.update(path.read_bytes())
        return stable_id(json.dumps(values, sort_keys=True) + source_hash.hexdigest() + self.ingestion.fingerprint(day))

    def process_day(self, day: str, force: bool = False) -> DayResult:
        key = self.cache_key(day)
        path = self.settings.paths.cache_dir / "days" / f"{day}-{key}.json"
        if path.exists() and not force:
            try:
                result = DayResult.model_validate_json(path.read_text(encoding="utf-8"))
                logger.info("%s 使用已核验缓存", day)
                docs, _, _ = self.ingestion.run(day)
                self.store.replace_day(day, result.observations, docs)
                self.store.record(result)
                return result
            except (ValueError, OSError):
                logger.warning("%s 缓存无效，重新运行", day)
        trace = []
        start = time.perf_counter()
        docs, batches, counts = self.ingestion.run(day)
        trace.append({"agent": "ingestion", "seconds": round(time.perf_counter()-start, 3), **counts})
        start = time.perf_counter()
        observations = self.extraction.run(docs, batches)
        fallbacks = sum(any(w.startswith("llm_fallback") for w in o.warnings) for o in observations)
        trace.append({"agent": "extraction", "seconds": round(time.perf_counter()-start, 3),
                      "observations": len(observations), "llm_fallbacks": fallbacks})
        start = time.perf_counter()
        observations = self.event_units.run(docs, observations, keep_documents=self.settings.alignment.mode == "one_to_one")
        trace.append({"agent": "event_units", "seconds": round(time.perf_counter()-start, 3),
                      "original_documents": len(docs), "event_observations": len(observations),
                      "split_units": sum(bool(o.metadata.get("parent_observation_id")) for o in observations)})
        start = time.perf_counter()
        # Canonical regions and PDF directions are persisted before retrieval.
        self.store.replace_day(day, observations, docs)
        observations = self.store.observations(day)
        candidates = self.alignment.run(observations, self.store.documents(day))
        previous = self.store.latest(day)
        previous_groups = {c.id: c for c in previous.candidates} if previous else {}
        for candidate in candidates:
            old = previous_groups.get(candidate.id)
            if old and old.observation_ids != candidate.observation_ids:
                candidate.assignment["previous_assignment"] = old.observation_ids
                candidate.assignment["change_basis"] = "按当前全部证据重算整体归属，未锁定旧对应"
        trace.append({"agent": "alignment", "seconds": round(time.perf_counter()-start, 3), "candidates": len(candidates),
                      "method": self.alignment.audit.get("method"), "rules": self.settings.alignment.model_dump(),
                      **self.alignment.audit})
        start = time.perf_counter()
        events = self.verification.run(candidates, observations)
        trace.append({"agent": "verification", "seconds": round(time.perf_counter()-start, 3),
                      "method": "two_source_verification_v1", "rules": self.settings.verification.model_dump(),
                      "classification": {s: sum(e.classification == s for e in events) for s in ("trusted", "suspicious", "unpaired")},
                      "accepted": sum(e.accepted for e in events), "rejected": sum(not e.accepted for e in events)})
        result = DayResult(date=day, observations=observations, candidates=candidates, events=events, trace=trace,
                           unassigned_observation_ids=self.alignment.unassigned_observation_ids)
        self.store.record(result)
        # 网络异常回退结果仍写审计，但不缓存为成功的模型运行，重跑可以恢复。
        if not fallbacks and not self.alignment.audit.get("llm_review_fallbacks"):
            write_json(path, result.model_dump(mode="json"))
        logger.info("%s 观测 %d，候选 %d，接纳 %d，模型回退 %d", day, len(observations), len(candidates), sum(e.accepted for e in events), fallbacks)
        return result

    def run(self, targets: list[str], force: bool = False) -> dict:
        available = self.ingestion.dates()
        if not targets:
            raise ValueError("没有指定日报日期")
        missing = sorted(set(targets)-set(available))
        if missing:
            raise ValueError("数据集中缺少日报日期：" + "、".join(missing))
        needed = set(targets)
        for day in targets:
            lower = date.fromisoformat(day)-timedelta(days=self.settings.report.lookback_days-1)
            needed.update(d for d in available if lower.isoformat() <= d <= day)
        out = self.settings.paths.output_dir
        days = []
        review_summary = None
        exporter = (ReviewExporter(self.settings.paths.data_dir, self.settings.paths.review_dir or out / "review",
                                   out / "daily") if self.settings.review_export.enabled else None)
        started_at = datetime.now(timezone.utc).isoformat()

        def checkpoint(status, current=None, error=None):
            write_json(out / "progress.json", {
                "status": status, "started_at": started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "target_dates": sorted(set(targets)), "needed_dates": sorted(needed),
                "completed_dates": [d.date for d in days], "completed_days": len(days),
                "total_days": len(needed), "current_date": current,
                "model_enabled": self.llm is not None,
                "model_name": (self.settings.llm.ollama.model if self.settings.llm.provider == "ollama"
                               else self.settings.llm.api.model) if self.llm else None,
                "strict_llm": self.settings.llm.fail_on_error,
                "llm_requests": self.llm.calls if self.llm else 0,
                "llm_cache_hits": self.llm.cache_hits if self.llm else 0,
                "error": error,
            })

        current = None
        try:
            for current in sorted(needed):
                checkpoint("running", current)
                logger.info("日期进度 %d/%d：%s", len(days) + 1, len(needed), current)
                result = self.process_day(current, force)
                write_json(out / "daily" / (current + ".json"), result.model_dump(mode="json"))
                # Earlier days in this sorted run cover the report's history window.
                if current in targets:
                    temporal = self.temporal.run(current, days + [result])
                    report, markdown = self.report.run(result, temporal)
                    write_json(out / "reports" / (current + ".json"), report)
                    write_text(out / "reports" / (current + ".md"), markdown)
                if exporter:
                    review_summary = exporter.export([result])
                days.append(result)
                checkpoint("running")
            write_json(out / "knowledge_graph.json", build_graph(days))
        except (Exception, KeyboardInterrupt) as exc:
            checkpoint("failed", current, f"{type(exc).__name__}: {exc}")
            raise
        if review_summary:
            review_summary["updated_dates"] = [d.date for d in days]
        fallback_count = sum(t.get("llm_fallbacks", 0) for d in days for t in d.trace)
        methods = sorted({t["method"] for d in days for t in d.trace if t.get("agent") == "alignment" and t.get("method")})
        manifest = {"version": __version__, "target_dates": sorted(set(targets)), "processed_dates": sorted(needed),
                    "model_enabled": self.llm is not None, "llm_requests": self.llm.calls if self.llm else 0,
                    "model_provider": self.settings.llm.provider if self.llm else None,
                    "model_name": (self.settings.llm.ollama.model if self.settings.llm.provider == "ollama"
                                   else self.settings.llm.api.model) if self.llm else None,
                    "lookback_days": self.settings.report.lookback_days,
                    "llm_cache_hits": self.llm.cache_hits if self.llm else 0, "llm_fallbacks": fallback_count,
                    "accepted_events": sum(e.accepted for d in days for e in d.events),
                    "event_classification": {s: sum(e.classification == s for d in days for e in d.events)
                                             for s in ("trusted", "suspicious", "unpaired")},
                    "association_method": methods[0] if len(methods) == 1 else "mixed",
                    "llm_review_fallbacks": sum(t.get("llm_review_fallbacks", 0) for d in days for t in d.trace),
                    "unassigned_observations": sum(len(d.unassigned_observation_ids) for d in days),
                    "evidence_database": str(self.settings.paths.output_dir / "evidence.sqlite"),
                    "association_states": {s: sum(c.relation_status == s for d in days for c in d.candidates)
                                           for s in ("confirmed", "candidate", "conflicted", "unmatched")},
                    "total_candidates": sum(len(d.candidates) for d in days), "output_dir": str(out),
                    "review_export": review_summary,
                    "format": "internal_v1_not_official_submission"}
        write_json(out / "manifest.json", manifest)
        checkpoint("complete")
        return manifest

