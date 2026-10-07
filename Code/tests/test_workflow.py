"""Behavioral tests for recovery, source separation and evidence preservation."""
import json
from types import SimpleNamespace
import pytest

from fusion_agent.config import Settings
from fusion_agent.domain import canonical_platform, platform_catalog
from fusion_agent.evidence_store import EvidenceStore
from fusion_agent.semantics import association_quote_options
from fusion_agent.workflow import EvidenceWorkflow
from fusion_agent.agents.workflow_agents import EntityResolutionAgent, SourceCompletenessAgent
from test_semantic_judgment import group, normal_review, semantic_text


class WorkflowModel:
    def __init__(self, rejected=(), recover=(), source_claims=None):
        self.settings = SimpleNamespace(fail_on_error=True)
        self.rejected, self.recover = set(rejected), set(recover)
        self.requests = []
        self.source_claims = source_claims or {}

    def complete_json(self, system, user, schema):
        request = json.loads(user)
        self.requests.append(request)
        if "source_recheck" in request:
            return {i: {field: {"status": "explicit" if (claims := self.source_claims.get((request["source_recheck"], field))) else "absent",
                               "claims": claims or [], "reason": "仅核对当前原文中该主体的明确陈述"}
                        for field in unit["needed_fields"]} for i, unit in request["units"].items()}
        if "recovery_candidates" in request:
            def check_enums(node):
                if isinstance(node, dict):
                    if "enum" in node:
                        assert node["enum"], "Ollama rejects empty JSON schema enums"
                    for value in node.values():
                        check_enums(value)
                elif isinstance(node, list):
                    for value in node:
                        check_enums(value)
            check_enums(schema)
            return {item["id"]: {"decision": "candidate" if item["id"] in self.recover else "unrelated",
                    "news_quote": association_quote_options(request["news_text"])[0],
                    "source_quote": association_quote_options(item["text"])[0] if item["text"] else "",
                    "row_numbers": sorted({n for row in item["radar_rows"] for n in row["row_numbers"]}),
                    "reason": "检查原始主体与活动上下文，提议进一步审查"}
                    for item in request["recovery_candidates"]}
        result = normal_review(request)
        if "opinion_candidates" in request:
            for decision in result["decisions"]:
                if decision["opinion_id"] in self.rejected:
                    decision["decision"] = "different_event"
        return result


def run_workflow(tmp_path, observations, docs, model, **options):
    settings = Settings()
    settings.workflow.enabled = True
    for name, value in options.items():
        if name == "semantic_candidates":
            settings.alignment.semantic_candidates = value
        else:
            setattr(settings.workflow, name, value)
    workflow = EvidenceWorkflow(settings, model)
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    try:
        result = workflow.run(observations, docs, store)
        assert [o.facts for o in store.observations(docs[0].date)] == [o.facts for o in result[0]]
        return workflow, result
    finally:
        store.close()


def test_cited_ship_name_variant_is_normalized_and_raw_model_value_is_retained():
    text = "美国海军“新奥尔良”号船坞运输舰于2025年5月1日在黄海执行编队警戒部署。"
    _, obs = semantic_text(text, {"country": [{"value": "美国", "quote": text}],
                                  "platform": [{"value": "新奥尔良", "quote": text}],
                                  "region": [{"value": "黄海", "quote": text}]}, strict=True)
    assert obs.facts["platform"] == ["新奥尔良号"]
    assert obs.metadata["llm_extraction"]["activities"][0]["platform"][0]["value"] == "新奥尔良"
    normalization = obs.metadata["entity_normalizations"][0]
    assert normalization["model_value"] == "新奥尔良" and normalization["quote"] == text


def test_ship_class_and_other_ship_names_are_not_merged():
    aliases = platform_catalog(["新奥尔良号船坞运输舰", "本福德号导弹驱逐舰"])
    assert canonical_platform("新奥尔良级", "新奥尔良级巡洋舰", aliases) == "新奥尔良级"
    assert canonical_platform("本福德", "新奥尔良号船坞运输舰", aliases) == "本福德"
    assert canonical_platform("新奥尔良", "新奥尔良级与新奥尔良号船坞运输舰", aliases) == "新奥尔良"
    assert canonical_platform("摩耶号", "摩耶号驱逐舰") == "摩耶号"
    assert canonical_platform("马尔瓦尔", "米格尔·马尔瓦尔号护卫舰") == "马尔瓦尔"


def test_ambiguous_short_ship_name_is_model_reviewed_and_named_identifiers_follow_resolution():
    docs, observations, _ = group()
    full_name, short_name = "米格尔·马尔瓦尔号", "马尔瓦尔"
    docs[0].text, docs[1].text = "菲律宾米格尔·马尔瓦尔号护卫舰在东海活动。", "菲律宾马尔瓦尔护卫舰在东海活动。"
    for obs, doc, name in zip(observations[:2], docs, [full_name, short_name]):
        obs.facts["platform"] = [name]
        obs.quotes["platform"] = [doc.text]
        obs.metadata["individual_ids"] = [name if name.endswith("号") else name + "号"]
    observations[2].facts["platform"] = [full_name]
    class EntityModel:
        def __init__(self):
            self.requests = []
        def complete_json(self, system, user, schema):
            self.requests.append(json.loads(user))
            return {"decision": "same_entity", "canonical_name": full_name, "quote": docs[1].text,
                    "reason": "原文名称为候选舰名的简称"}
    model = EntityModel()
    resolved = EntityResolutionAgent(model).run(observations, docs)
    assert len(model.requests) == 1  # Radar must not hide the text representative.
    assert resolved[1].facts["platform"] == [full_name]
    assert resolved[1].metadata["individual_ids"] == [full_name]
    assert observations[1].facts["platform"] == [short_name]


def test_recovery_can_find_candidate_outside_initial_review_window(tmp_path):
    docs, observations, _ = group()
    initial = observations[1]
    initial.id, initial.path = "p1", "001.txt"
    docs[1].id, docs[1].path = initial.id, initial.path
    other = initial.model_copy(deep=True)
    other.id, other.path = "p2", "002.txt"
    other_doc = docs[1].model_copy(update={"id": other.id, "path": other.path})
    observations.append(other)
    docs.append(other_doc)
    model = WorkflowModel(rejected={"p1"}, recover={"p2"})
    workflow, (_, candidates, events, trace) = run_workflow(tmp_path, observations, docs, model, semantic_candidates=1)
    assert candidates[0].assignment["opinion_id"] == "p2"
    assert events[0].classification == "trusted"
    assert workflow.audit["recovered_groups"] == 1
    assert candidates[0].assignment["opinion_candidates"][0]["llm_review"]["decision"] == "same_event"
    assert any(r["decision"] == "different_event" for r in candidates[0].assignment["semantic_link_reviews"]["opinions"])
    assert any(t["to"] == "candidate_recall" and t["reason"] == "reviewer_requested_new_evidence"
               for t in workflow.audit["transitions"])
    assert any(t["agent"] == "exception_review" for t in trace)
    assert sum("claims" in request for request in model.requests) == 2  # New source requires a new fact review.

    # Reusing the controller for another run must not carry over recovery
    # decisions or retrieval routes from the previous evidence pool.
    next_docs, next_observations, _ = group()
    next_store = EvidenceStore(tmp_path / "next.sqlite")
    try:
        workflow.run(next_observations, next_docs, next_store)
        assert workflow.audit["recovery_reviews"] == []
        assert not any(oid == "p2" for _, oid in workflow.recall.routes)
    finally:
        next_store.close()


def test_recovery_proposal_is_not_itself_an_activity_confirmation(tmp_path):
    docs, observations, _ = group()
    second = observations[1].model_copy(deep=True)
    second.id, second.path = "p2", "p2.txt"
    docs.append(docs[1].model_copy(update={"id": "p2", "path": "p2.txt"}))
    observations.append(second)
    model = WorkflowModel(rejected={"p", "p2"}, recover={"p2"})
    workflow, (_, candidates, events, _) = run_workflow(tmp_path, observations, docs, model, semantic_candidates=1)
    assert candidates[0].assignment["opinion_id"] is None
    assert events[0].classification == "untrusted"
    assert workflow.audit["recovery_proposals"] == 1 and workflow.audit["recovered_groups"] == 0
    assert sum("claims" in request for request in model.requests) == 1
    assert workflow.audit["verification_reuses"] == 1  # Rejected proposal leaves selected evidence unchanged.


def test_budget_exhaustion_records_incomplete_search_without_forcing_a_pair(tmp_path):
    docs, observations, _ = group()
    second = observations[1].model_copy(update={"id": "p2", "path": "p2.txt"}, deep=True)
    docs.append(docs[1].model_copy(update={"id": "p2", "path": "p2.txt"}))
    observations.append(second)
    model = WorkflowModel(rejected={"p"})
    workflow, (_, candidates, events, _) = run_workflow(tmp_path, observations, docs, model,
                                                       semantic_candidates=1, max_recovery_rounds=0)
    assert workflow.audit["stop_reason"] == "recovery_budget_exhausted"
    assert candidates[0].assignment["opinion_id"] is None
    diagnostic = events[0].semantic_review["workflow_diagnostics"]
    assert diagnostic["source_coverage"] == "missing_source"
    assert not diagnostic["search_coverage"]["opinion"]["complete"]
    assert events[0].classification == "untrusted"


def test_real_fact_conflict_is_retained_and_does_not_trigger_search_for_agreement(tmp_path):
    docs, observations, _ = group()
    observations[1].facts["quantity"] = [9]
    original = [o.model_dump() for o in observations]
    workflow, (_, _, events, _) = run_workflow(tmp_path, observations, docs, WorkflowModel())
    assert events[0].classification == "untrusted"
    assert events[0].verdicts["quantity"].status == "conflict"
    assert events[0].semantic_review["workflow_diagnostics"]["facts"] == "conflict"
    assert workflow.audit["recovery_proposals"] == 0
    assert original == [o.model_dump() for o in observations]


def test_actual_missing_source_remains_unknown_and_untrusted(tmp_path):
    docs, observations, _ = group()
    docs = [d for d in docs if d.source != "opinion"]
    observations = [o for o in observations if o.source != "opinion"]
    _, (_, _, events, _) = run_workflow(tmp_path, observations, docs, WorkflowModel())
    diagnostic = events[0].semantic_review["workflow_diagnostics"]
    assert diagnostic["source_coverage"] == "missing_source"
    assert diagnostic["facts"] == "supported"
    assert events[0].classification == "untrusted"
    assert "three_sources_not_linked" in events[0].verification_reasons


def test_raw_source_recall_recovers_model_omitted_subject_without_inventing_a_fact(tmp_path):
    docs, observations, _ = group()
    observations[1].facts.pop("platform")
    observations[1].facts.pop("start")
    observations[1].facts.pop("end")
    observations[1].facts["region"] = ["东海"]
    from fusion_agent.agents.one_to_one import OneToOneAlignmentAgent
    assert OneToOneAlignmentAgent(Settings().alignment).pair_score(observations[0], observations[1]) == 0
    model = WorkflowModel()
    _, (resolved, candidates, events, _) = run_workflow(tmp_path, observations, docs, model)
    assert candidates[0].assignment["opinion_id"] == "p"
    assert "raw_subject_mention" in candidates[0].assignment["workflow"]["recall_routes"]["p"]
    assert not next(o for o in resolved if o.id == "p").facts.get("platform")
    assert events[0].classification == "trusted"


def test_source_local_review_restores_omitted_actor_country_with_citation():
    text = "法国一艘大邱级导弹护卫舰在巴伦支海活动。"
    doc, obs = semantic_text(text, {"platform": [{"value": "大邱级", "quote": text}],
                                  "region": [{"value": "巴伦支海", "quote": text}]})
    original = obs.model_dump()
    model = WorkflowModel(source_claims={(obs.id, "country"): [{"value": "法国", "quote": text}]})
    agent = SourceCompletenessAgent(model)
    result = agent.run([obs], [doc])[0]
    assert result.facts["country"] == ["法国"]
    assert result.metadata["semantic_activities"][0]["facts"]["country"] == ["法国"]
    assert result.quotes["country"] == [text]
    assert result.metadata["llm_extraction"] == obs.metadata["llm_extraction"]
    assert obs.model_dump() == original
    assert agent.audit["added_claims"] == 1
    assert all("sources" not in r and "source_text" in r for r in model.requests)


def test_source_review_does_not_union_media_country_or_clear_existing_direction():
    text = "美国媒体报道，菲律宾米格尔·马尔瓦尔号在上海外海活动。"
    doc, obs = semantic_text(text, {"country": [{"value": "菲律宾", "quote": text}],
                                  "platform": [{"value": "米格尔·马尔瓦尔号", "quote": text}],
                                  "region": [{"value": "上海外海", "quote": text}]})
    result = SourceCompletenessAgent(WorkflowModel()).run([obs], [doc])[0]
    assert result.facts["country"] == ["菲律宾"]
    assert result.facts["direction"] == obs.facts["direction"] == ["东海"]


def test_cited_aircraft_type_corrects_ship_metadata_without_rewriting_model_output():
    text = "中国2架歼-15T舰载战斗机在济州岛西部海域执行起降训练。"
    doc, obs = semantic_text(text, {"country": [{"value": "中国", "quote": text}],
                                  "platform": [{"value": "歼-15T", "quote": text}]}, strict=True)
    assert obs.metadata["platform_kinds"] == ["ship"]
    result = EntityResolutionAgent().run([obs], [doc])[0]
    assert result.metadata["platform_kinds"] == ["air"]
    assert result.metadata["semantic_activities"][0]["metadata"]["platform_kinds"] == ["air"]
    assert result.metadata["llm_extraction"]["activities"][0]["platform_kind"] == "ship"
    assert result.metadata["entity_type_resolutions"][0]["quote"] == text


def test_source_country_recheck_rejects_a_real_quote_that_does_not_support_the_value():
    text = "法国一艘大邱级护卫舰在巴伦支海活动。此次训练按计划完成。"
    doc, obs = semantic_text(text, {"platform": [{"value": "大邱级", "quote": text}]})
    model = WorkflowModel(source_claims={(obs.id, "country"): [{"value": "法国", "quote": "此次训练按计划完成。"}]})
    with pytest.raises(ValueError, match="值未由该条引用支持"):
        SourceCompletenessAgent(model).run([obs], [doc])
