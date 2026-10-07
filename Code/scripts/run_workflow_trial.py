"""Run the new workflow on fixed, source-validated extracted observations.

This isolates execution changes from extraction sampling. All new activity,
entity ambiguity, verification and exception decisions use the configured LLM.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code/src"))
from fusion_agent.config import load_settings
from fusion_agent.knowledge import build_graph
from fusion_agent.models import DayResult
from fusion_agent.pipeline import AgentPipeline
from fusion_agent.review_export import ReviewExporter
from fusion_agent.utils import write_json, write_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Code/outputs/agent_workflow_20250501")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    baseline_path = args.baseline or out / "baseline.json"
    baseline = DayResult.model_validate_json(baseline_path.read_text(encoding="utf-8-sig"))
    settings = load_settings(ROOT / "Code/config.toml")
    settings.paths.output_dir, settings.paths.review_dir = out, out / "review"
    settings.workflow.enabled, settings.llm.enabled, settings.llm.fail_on_error = True, True, True
    settings.report.lookback_days, settings.llm.max_calls = 1, 600
    pipeline = AgentPipeline(settings)
    started = time.perf_counter()
    # A repeat run must not expose old round snapshots as current progress.
    old_rounds = list((out / "rounds").glob("round_*.json"))
    if old_rounds:
        archive = out / "attempts" / ("rounds_" + str(time.time_ns()))
        archive.mkdir(parents=True, exist_ok=True)
        for previous_round in old_rounds:
            destination = archive / previous_round.name
            if not previous_round.resolve().is_relative_to(out) or not destination.resolve().is_relative_to(out):
                raise ValueError("轮次记录归档路径必须位于指定试验目录内")
            previous_round.rename(destination)
    if (out / "progress.json").exists():
        previous = json.loads((out / "progress.json").read_text(encoding="utf-8"))
        if previous.get("status") == "failed":
            write_json(out / "attempts" / ("failed_" + str(time.time_ns()) + ".json"), previous)
    write_json(out / "progress.json", {"status": "running", "input_mode": "saved_validated_extraction", "date": baseline.date})
    try:
        docs, _, counts = pipeline.ingestion.run(baseline.date)
        exporter = ReviewExporter(settings.paths.data_dir, settings.paths.review_dir, out / "daily")
        # Verify every source quote and every radar row against original data.
        exporter.preflight(baseline)
        def save_round(snapshot):
            write_json(out / "rounds" / ("round_" + str(snapshot["round"]) + ".json"), snapshot)
            write_json(out / "progress.json", {"status": "running", "input_mode": "saved_validated_extraction",
                       "date": baseline.date, "completed_round": snapshot["round"], "summary": snapshot["summary"],
                       "llm_requests": pipeline.llm.calls, "llm_cache_hits": pipeline.llm.cache_hits})
        observations, candidates, events, trace = pipeline.workflow.run(baseline.observations, docs, pipeline.store, on_round=save_round)
        result = DayResult(date=baseline.date, observations=observations, candidates=candidates, events=events,
                           trace=[{"agent": "ingestion", **counts}, {"agent": "extraction", "method": "saved_validated_extraction",
                                   "baseline": str(baseline_path), "observations": len(observations), "llm_fallbacks": 0}] + trace,
                           unassigned_observation_ids=pipeline.workflow.unassigned_observation_ids)
        pipeline.store.record(result)
        write_json(out / "daily" / (result.date + ".json"), result.model_dump(mode="json"))
        temporal = pipeline.temporal.run(result.date, [result])
        report, markdown = pipeline.report.run(result, temporal)
        write_json(out / "reports" / (result.date + ".json"), report)
        write_text(out / "reports" / (result.date + ".md"), markdown)
        exporter.export([result])
        write_json(out / "knowledge_graph.json", build_graph([result]))
        manifest = {"input_mode": "saved_validated_extraction", "extraction_baseline": str(baseline_path),
                    "extraction_baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                    "model_enabled": True, "agent_workflow_enabled": True, "model_name": pipeline.llm.settings.endpoint.model,
                    "model_provider": pipeline.llm.settings.provider, "strict_llm": True,
                    "target_dates": [result.date], "processed_dates": [result.date],
                    "llm_requests": pipeline.llm.calls, "llm_cache_hits": pipeline.llm.cache_hits,
                    "llm_validation_repairs": pipeline.llm.validation_repairs,
                    "llm_fallbacks": 0, "llm_review_fallbacks": next(t["llm_review_fallbacks"] for t in trace if t["agent"] == "alignment"),
                    "llm_verification_fallbacks": next(t["llm_verification_fallbacks"] for t in trace if t["agent"] == "verification"),
                    "event_classification": {s: sum(e.classification == s for e in events) for s in ("trusted", "untrusted")},
                    "accepted_events": sum(e.accepted for e in events), "elapsed_seconds": round(time.perf_counter()-started, 3),
                    "output_dir": str(out)}
        write_json(out / "manifest.json", manifest)
        write_json(out / "progress.json", {"status": "complete", **manifest})
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception as exc:
        write_json(out / "progress.json", {"status": "failed", "error": str(exc), "input_mode": "saved_validated_extraction",
                   "llm_requests": pipeline.llm.calls, "llm_cache_hits": pipeline.llm.cache_hits,
                   "llm_validation_repairs": pipeline.llm.validation_repairs,
                   "elapsed_seconds": round(time.perf_counter()-started, 3)})
        raise
    finally:
        pipeline.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
