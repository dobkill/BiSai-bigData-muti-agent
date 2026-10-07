import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import httpx
import pytest

from fusion_agent import cli
from fusion_agent.config import AdaptiveConcurrencySettings, LLMSettings, Paths, Settings
from fusion_agent.llm import LLMClient
from fusion_agent.runtime import add_llm_arguments, apply_llm_arguments, estimate_request_budget
from test_pipeline import dataset
from semantic_fixture import fixture_reply


def arguments(*values):
    parser = argparse.ArgumentParser()
    add_llm_arguments(parser)
    return parser.parse_args(values)


def test_backend_switch_preserves_all_endpoint_settings():
    settings = Settings(llm=LLMSettings(provider="llama_cpp", concurrency=4,
                                       adaptive=AdaptiveConcurrencySettings(enabled=True)))
    settings.llm.ollama.model = "old-local-model"
    settings.llm.api.model = "remote-model"
    apply_llm_arguments(settings, arguments("--provider", "api", "--concurrency", "8"))
    assert settings.llm.endpoint.model == "remote-model"
    assert not settings.llm.adaptive.enabled
    assert settings.llm.concurrency == 8
    apply_llm_arguments(settings, arguments("--provider", "ollama"))
    assert settings.llm.endpoint.model == "old-local-model"
    apply_llm_arguments(settings, arguments("--provider", "llama_cpp", "--adaptive-concurrency",
                                          "--model", "my-alias", "--base-url", "http://localhost:8081/v1"))
    assert settings.llm.endpoint.model == "my-alias" and settings.llm.adaptive.enabled
    assert settings.llm.ollama.model == "old-local-model"
    assert settings.llm.api.model == "remote-model"


@pytest.mark.parametrize("args", [("--concurrency", "0"), ("--provider", "api", "--adaptive-concurrency")])
def test_invalid_runtime_override_is_rejected(args):
    with pytest.raises(ValueError):
        apply_llm_arguments(Settings(), arguments(*args))


def test_full_cli_budgets_all_dates_and_resumes_llama_cpp(tmp_path, monkeypatch, capsys):
    root = tmp_path / "data"
    dataset(root)
    opinion = root / "2025.5.10/网上舆论/001.txt"
    opinion.write_text(opinion.read_text(encoding="utf-8") + "\n独立舆论来源。", encoding="utf-8")
    shutil.copytree(root / "2025.5.10", root / "2025.5.11")
    settings = Settings(paths=Paths(data_dir=root, output_dir=tmp_path / "out", cache_dir=tmp_path / "cache"))
    settings.alignment.max_llm_reviews = 0
    settings.llm.max_calls = 1  # would stop a full run without the automatic budget
    settings.llm.retries = 1
    requests = []
    original_init = LLMClient.__init__

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"total_slots": 2})
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(fixture_reply(requests[-1]), ensure_ascii=False)}}]})

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.http.close()
        self.http = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(LLMClient, "__init__", init)
    monkeypatch.setattr(cli, "load_settings", lambda path: settings)
    monkeypatch.setattr(sys, "argv", ["fusion_agent", "run", "--all", "--use-llm", "--strict-llm",
                                    "--provider", "llama_cpp", "--concurrency", "4", "--fixed-concurrency"])
    cli.main()
    first = json.loads(capsys.readouterr().out)
    assert first["processed_dates"] == ["2025-05-10", "2025-05-11"]
    assert first["model_provider"] == "llama_cpp" and first["model_name"] == "local-model"
    assert first["llm_requests"] == 8 and first["llm_fallbacks"] == 0
    assert first["llm_verifications"] == 2 and first["decision_engine"] == "llm_semantic"
    assert first["llm_concurrency"]["maximum"] == 2
    assert settings.llm.max_calls == estimate_request_budget(settings, first["processed_dates"]) == 16
    assert (tmp_path / "out/reports/2025-05-11.md").is_file()
    cli.main()
    second = json.loads(capsys.readouterr().out)
    assert second["llm_requests"] == 0
    assert len(requests) == 8


def test_explicit_full_budget_is_preserved(tmp_path, monkeypatch, capsys):
    root = tmp_path / "data"
    dataset(root)
    settings = Settings(paths=Paths(data_dir=root, output_dir=tmp_path / "out", cache_dir=tmp_path / "cache"))
    seen = {}

    class Pipeline:
        def __init__(self, effective):
            seen["max_calls"] = effective.llm.max_calls

        def run(self, targets, force=False):
            return {"targets": targets}

        def close(self):
            pass

    monkeypatch.setattr(cli, "load_settings", lambda p: settings)
    monkeypatch.setattr(cli, "AgentPipeline", Pipeline)
    monkeypatch.setattr(sys, "argv", ["fusion_agent", "run", "--all", "--use-llm", "--max-llm-calls", "3"])
    cli.main()
    capsys.readouterr()
    assert seen["max_calls"] == 3


def test_full_workflow_budget_accounts_for_recovery_and_entity_review(tmp_path):
    root = tmp_path / "data"
    dataset(root)
    settings = Settings(paths=Paths(data_dir=root, output_dir=tmp_path / "out", cache_dir=tmp_path / "cache"))
    targets = ["2025-05-10"]
    old = estimate_request_budget(settings, targets)
    settings.workflow.enabled = True
    settings.workflow.max_recovery_rounds = 2
    with_recovery = estimate_request_budget(settings, targets)
    assert with_recovery > old
    settings.workflow.max_recovery_rounds = 0
    assert old < estimate_request_budget(settings, targets) < with_recovery
    settings.alignment.mode = "legacy"
    settings.workflow.enabled = False
    legacy = estimate_request_budget(settings, targets)
    settings.workflow.enabled = True
    assert estimate_request_budget(settings, targets) == legacy


def test_cli_runs_agent_workflow_and_exports_state_trace(tmp_path, monkeypatch, capsys):
    root = tmp_path / "data"
    dataset(root)
    out = tmp_path / "workflow"
    settings = Settings(paths=Paths(data_dir=root, output_dir=out, cache_dir=tmp_path / "cache"))
    settings.llm.ollama.model = "fixture-model"
    original_init = LLMClient.__init__

    def handler(request):
        reply = fixture_reply(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": json.dumps(reply, ensure_ascii=False)}})

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.http.close()
        self.http = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(LLMClient, "__init__", init)
    monkeypatch.setattr(cli, "load_settings", lambda p: settings)
    monkeypatch.setattr(sys, "argv", ["fusion_agent", "run", "--date", "2025-05-10", "--single-day",
                                    "--use-llm", "--strict-llm", "--agent-workflow", "--output-dir", str(out)])
    cli.main()
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["agent_workflow_enabled"]
    assert manifest["llm_requests"] > 0 and manifest["llm_fallbacks"] == 0
    assert manifest["llm_verifications"] == 1
    day = json.loads((out / "daily/2025-05-10.json").read_text(encoding="utf-8"))
    trace = next(t for t in day["trace"] if t["agent"] == "workflow")
    assert next(t for t in day["trace"] if t["agent"] == "alignment")["llm_validated_reviews"] == 1
    assert trace["transitions"][0]["to"] == "source_completeness"
    assert trace["transitions"][-1]["to"] == "output"
    assert day["events"][0]["semantic_review"]["workflow_diagnostics"]["source_coverage"] == "complete"
    assert (out / "review/2025-05-10/README.md").is_file()


def test_launcher_enables_batching_and_preserves_per_slot_context(monkeypatch, tmp_path):
    script = Path(__file__).parents[1] / "scripts/start_llama_server.py"
    spec = importlib.util.spec_from_file_location("start_llama_server", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = Settings()
    settings.llm.llama_cpp.api_key = "never-print-this"
    command = module.server_command(settings, "llama-server.exe", tmp_path / "model.gguf", slots=2)
    assert command[command.index("--parallel") + 1] == "2"
    assert command[command.index("--ctx-size") + 1] == "16384"
    assert "--cont-batching" in command and "--no-context-shift" in command
    assert "never-print-this" not in json.dumps(command)


def test_full_script_selected_backend_validates_every_day_and_redacts_keys(tmp_path, monkeypatch, capsys):
    script = Path(__file__).parents[1] / "scripts/run_full_local.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("run_full_local", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "data"
    dataset(root)
    shutil.copytree(root / "2025.5.10", root / "2025.5.11")
    out, review = tmp_path / "full", tmp_path / "review"
    settings = Settings(paths=Paths(data_dir=root, output_dir=out, cache_dir=tmp_path / "cache"))
    settings.alignment.max_llm_reviews = 0
    settings.llm.llama_cpp.api_key = "llama-secret"
    settings.llm.api.api_key = "api-secret"
    original_inspect = module.inspect_data

    def inspect(effective, path):
        inventory = original_inspect(effective, path)
        # This deliberately small fixture exercises the complete workflow with two texts/day.
        inventory["nonstandard_days"] = []
        return inventory

    original_init = LLMClient.__init__

    def handler(request):
        assert request.headers["Authorization"] == "Bearer llama-secret"
        if request.method == "GET":
            return httpx.Response(200, json={"total_slots": 2})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(fixture_reply(json.loads(request.content)), ensure_ascii=False)}}]})

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.http.close()
        self.http = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(LLMClient, "__init__", init)
    monkeypatch.setattr(module, "load_settings", lambda p: settings)
    monkeypatch.setattr(module, "inspect_data", inspect)
    monkeypatch.setattr(sys, "argv", [str(script), "--provider", "llama_cpp", "--fixed-concurrency", "--concurrency", "2",
                                    "--output-dir", str(out), "--review-dir", str(review)])
    module.main()
    job = json.loads(capsys.readouterr().out)
    assert job["status"] == "complete" and job["validated_days"] == 2
    assert job["provider"] == "llama_cpp" and job["model"] == "local-model"
    assert job["llm_fallbacks"] == 0 and job["llm_requests"] == 6
    validations = json.loads((out / "structure_validation.json").read_text(encoding="utf-8"))
    assert [v["date"] for v in validations] == ["2025-05-10", "2025-05-11"]
    for name in ("effective_settings.json", "full_run.json", "progress.json", "manifest.json"):
        text = (out / name).read_text(encoding="utf-8")
        assert "llama-secret" not in text and "api-secret" not in text
