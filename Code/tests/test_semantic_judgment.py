import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from fusion_agent.agents.event_units import EventUnitAgent
from fusion_agent.agents.extraction import ExtractionAgent
from fusion_agent.agents.one_to_one import OneToOneAlignmentAgent
from fusion_agent.agents.verification import VerificationAgent
from fusion_agent.config import LLMSettings, Paths, ReportSettings, Settings, VerificationSettings
from fusion_agent.domain import regions
from fusion_agent.llm import LLMClient
from fusion_agent.models import Candidate, Document, RadarBatch
from fusion_agent.pipeline import AgentPipeline
from semantic_fixture import fixture_reply
from test_pipeline import dataset


class ModelReply:
    def __init__(self, response, strict=True):
        self.response = response
        self.settings = SimpleNamespace(fail_on_error=strict)
        self.requests = []

    def complete_json(self, system, user, schema):
        self.requests.append((system, user, schema))
        return self.response(json.loads(user)) if callable(self.response) else deepcopy(self.response)


def semantic_text(text, fields, strict=False):
    doc = Document(id="n", date="2025-05-01", source="open_source", path="n.txt", text=text)
    reply = {"activities": [{"activity_id": "one", "platform_kind": "ship", **fields}]}
    settings = Settings()
    settings.llm.fail_on_error = strict
    return doc, ExtractionAgent(settings, ModelReply(reply)).text(doc)


def group(counts=(1, 1), region="东海中部"):
    docs = [Document(id=oid, date="2025-05-01", source=source, path=oid + ".txt",
                     text=f"2025年5月1日，中国1艘054A型护卫舰在{region}从10时00分至12时00分执行战备巡逻。")
            for oid, source in (("n", "open_source"), ("p", "opinion"))]
    rows = [{"国家地区": "中国", "机舰名": "054A型护卫舰", "数量": q, "活动区域": region,
             "首点时间": "2025.05.01 10:00:00", "结束时间": "2025.05.01 12:00:00", "任务": "战备巡逻"}
            for q in counts]
    extractor = ExtractionAgent(Settings(), None)
    radar = extractor.radar(RadarBatch(id="r", date="2025-05-01", path="r.xlsx", sheet="舰船",
                                      batch="SHIP-1", rows=rows, row_numbers=list(range(2, len(rows) + 2))))
    observations = [extractor.text(d) for d in docs] + [radar]
    candidate = Candidate(id="event", date="2025-05-01", observation_ids=[o.id for o in observations],
                          relation_status="confirmed")
    return docs, observations, candidate


def normal_review(request):
    return fixture_reply({"messages": [{"content": json.dumps(request, ensure_ascii=False)}]})


def verify(response, counts=(1, 1), strict=True):
    docs, observations, candidate = group(counts)
    model = ModelReply(response, strict)
    verifier = VerificationAgent(VerificationSettings(), llm=model)
    event = verifier.run([candidate], observations, docs)[0]
    return event, verifier, model


def test_navy_name_is_not_a_sea_but_real_philippine_sea_remains():
    assert regions("菲律宾海军在上海外海活动") == ["上海外海"]
    assert regions("菲律宾海军舰艇在菲律宾海活动") == ["菲律宾海"]


def test_model_selects_subject_country_and_location_without_rule_union():
    text = "美国媒体报道，菲律宾海军米格尔·马尔瓦尔号在上海外海活动。"
    _, obs = semantic_text(text, {"country": [{"value": "菲律宾", "quote": text}],
                                  "platform": [{"value": "米格尔·马尔瓦尔号", "quote": text}],
                                  "region": [{"value": "上海外海", "quote": text}]})
    assert obs.facts["country"] == ["菲律宾"]
    assert obs.facts["region"] == ["上海外海"] and obs.facts["direction"] == ["东海"]
    assert "美国" in obs.metadata["rule_candidates"]["country"]
    assert obs.metadata["extractor"] == "llm_semantic"


def test_model_omitted_region_is_not_silently_supplied_by_keyword():
    _, obs = semantic_text("中国054A型护卫舰在南海活动。", {"platform": [{"value": "054A", "quote": "054A型护卫舰"}]})
    assert not obs.facts.get("region") and obs.metadata["rule_candidates"]["region"] == ["南海"]


def test_bad_selected_value_is_rejected_without_being_replaced_by_regex_value():
    _, obs = semantic_text("中国054A型护卫舰在东海活动。", {"country": [{"value": "美国", "quote": "中国"}]})
    assert not obs.facts.get("country")
    assert any(w.startswith("llm_unvalidated_claim:country") for w in obs.warnings)
    with pytest.raises(ValueError, match="引用校验失败"):
        semantic_text("中国054A型护卫舰在东海活动。", {"country": [{"value": "美国", "quote": "中国"}]}, strict=True)


def test_publication_and_background_times_do_not_override_model_activity_time():
    text = "发布于2025年5月2日。正文：2025年5月1日，中国054A型护卫舰从10时00分至12时00分在东海活动。背景：此前从03时00分至04时00分。"
    quote = "2025年5月1日，中国054A型护卫舰从10时00分至12时00分在东海活动"
    doc, obs = semantic_text(text, {"platform": [{"value": "054A", "quote": quote}],
                                  "start": [{"value": "2025-05-01T10:00:00", "quote": quote}],
                                  "end": [{"value": "2025-05-01T12:00:00", "quote": quote}]})
    canonical = EventUnitAgent().run([doc], [obs], keep_documents=True)[0]
    assert canonical.facts["start"] == ["2025-05-01T10:00:00"]
    assert canonical.facts["end"] == ["2025-05-01T12:00:00"]
    assert "text_date_mismatch" not in canonical.warnings


def test_model_cannot_invent_a_different_date_from_date_free_time_quote():
    _, obs = semantic_text("中国054A从10时00分至12时00分在东海活动。",
                           {"start": [{"value": "2025-05-02T10:00:00", "quote": "从10时00分至12时00分"}]})
    assert not obs.facts.get("start")
    assert any(w.startswith("llm_unvalidated_claim:start") for w in obs.warnings)


def test_activity_scopes_preserve_separate_quantities_and_do_not_cross_time_windows():
    text = "中国1艘054A从10时00分至12时00分在东海活动；日本2艘摩耶级从15时00分至17时00分在黄海活动。"
    activities = []
    for i, (name, q, start, end, quote) in enumerate((
            ("054A", 1, "10:00", "12:00", text.split("；")[0]),
            ("摩耶级", 2, "15:00", "17:00", text.split("；")[1]))):
        activities.append({"activity_id": str(i), "platform_kind": "ship",
                           "platform": [{"value": name, "quote": quote}],
                           "quantity": [{"value": q, "quote": quote}],
                           "start": [{"value": "2025-05-01T" + start + ":00", "quote": quote}],
                           "end": [{"value": "2025-05-01T" + end + ":00", "quote": quote}]})
    doc = Document(id="n", date="2025-05-01", source="open_source", path="n.txt", text=text)
    obs = ExtractionAgent(Settings(), ModelReply({"activities": activities})).text(doc)
    units = EventUnitAgent().run([doc], [obs], keep_documents=True)[0].metadata["platform_units"]
    assert [u["facts"]["quantity"] for u in units] == [[1], [2]]
    assert "invalid_time_interval" not in obs.warnings


def test_unambiguous_pairs_and_radar_are_also_reviewed_by_model():
    docs, observations, _ = group()
    model = ModelReply(normal_review)
    agent = OneToOneAlignmentAgent(Settings().alignment, model)
    candidates = agent.run(observations, docs)
    assert len(model.requests) == 1 and agent.audit["llm_validated_reviews"] == 1
    assert candidates[0].assignment["radar_assignments"][0]["llm_review"]["decision"] == "same_event"


def test_model_can_reject_a_pair_that_rules_rank_highly():
    docs, observations, _ = group()
    def reject(request):
        result = normal_review(request)
        result["decisions"][0]["decision"] = "different_event"
        return result
    agent = OneToOneAlignmentAgent(Settings().alignment, ModelReply(reject))
    candidate = agent.run(observations, docs)[0]
    assert "p" not in candidate.observation_ids and "p" in agent.unassigned_observation_ids
    assert candidate.assignment["opinion_candidates"][0]["llm_review"]["decision"] == "different_event"


def test_conflicting_fields_do_not_filter_subject_out_before_model_review():
    docs, observations, _ = group()
    observations[1].facts.update(country=["意大利"], direction=["南海"], region=["南海"], task=["远海训练"], quantity=[9])
    observations[1].facts.pop("start")
    observations[1].facts.pop("end")
    model = ModelReply(normal_review)
    agent = OneToOneAlignmentAgent(Settings().alignment, model)
    assert agent.pair_score(observations[0], observations[1]) < Settings().alignment.minimum_pair_score
    candidate = agent.run(observations, docs)[0]
    assert candidate.assignment["opinion_id"] == "p"
    assert "p" in {r["opinion_id"] for r in json.loads(model.requests[0][1])["opinion_candidates"]}


def test_final_review_receives_raw_sources_and_records_decision_audit():
    event, verifier, model = verify(normal_review)
    assert event.classification == "trusted" and event.accepted
    assert verifier.audit["llm_verifications"] == 1
    request = json.loads(model.requests[0][1])
    assert request["sources"][0]["text"] and request["sources"][2]["radar_rows"]
    assert event.semantic_review["status"] == "validated"


def test_final_model_activity_rejection_blocks_matching_field_values():
    def reject(request):
        result = normal_review(request)
        result["association"] = "different_activity"
        return result
    event, _, _ = verify(reject)
    assert event.classification == "untrusted"
    assert "semantic_association_not_confirmed:different_activity" in event.reasons


def test_model_cannot_majority_vote_away_radar_count_noise():
    def approve(request):
        result = normal_review(request)
        f = next(f for f in result["fields"] if f["field"] == "quantity")
        f.update(status="supported", value=1)
        return result
    event, _, _ = verify(approve, counts=(1, 2))
    assert event.classification == "untrusted" and event.verdicts["quantity"].status == "conflict"
    assert "retained_explicit_conflict:quantity" in event.semantic_review["code_checks"]


@pytest.mark.parametrize("action", ["omit", "exclude", "invent"])
def test_model_cannot_hide_or_invent_evidence(action):
    def change(request):
        result = normal_review(request)
        f = next(f for f in result["fields"] if f["field"] == "quantity")
        cid = next(c["claim_id"] for c in request["claims"] if c["source"] == "radar" and c["field"] == "quantity" and c["value"] == 2)
        if action in {"omit", "exclude"}:
            f["claim_ids"].remove(cid)
        if action == "exclude":
            result["excluded_claims"] = [{"claim_id": cid, "role": "background", "reason": "想去掉雷达噪声"}]
        if action == "invent":
            f["claim_ids"].append("imaginary-evidence")
        return result
    with pytest.raises(ValueError):
        verify(change, counts=(1, 2))


def test_unanchored_review_is_untrusted_even_when_offline_rules_would_accept():
    def hallucinate(request):
        result = normal_review(request)
        result["association_evidence"][0]["quote"] = "不存在的原文"
        return result
    event, verifier, _ = verify(hallucinate, strict=False)
    assert event.classification == "untrusted" and "semantic_verification_failed" in event.reasons
    assert verifier.audit["llm_verification_fallbacks"] == 1


def test_broad_sea_support_does_not_prove_single_source_specific_region():
    docs, observations, candidate = group(region="东海")
    observations[-1].facts["region"] = ["上海外海"]
    def overspecify(request):
        result = normal_review(request)
        f = next(f for f in result["fields"] if f["field"] == "region")
        f.update(status="supported", value="上海外海")
        return result
    event = VerificationAgent(VerificationSettings(), llm=ModelReply(overspecify)).run([candidate], observations, docs)[0]
    assert event.facts["region"] == "东海"


def test_invalid_review_cache_is_removed_and_retry_recovers(tmp_path):
    dataset(tmp_path / "data")
    settings = Settings(paths=Paths(data_dir=tmp_path / "data", output_dir=tmp_path / "out", cache_dir=tmp_path / "cache"),
                        report=ReportSettings(lookback_days=1),
                        llm=LLMSettings(enabled=True, ollama={"model": "fixture"}, retries=0, fail_on_error=False))
    pipeline = AgentPipeline(settings)
    pipeline.llm.http.close()
    state = {"invalid": True}
    def handler(request):
        payload = json.loads(request.content)
        result = fixture_reply(payload)
        if state["invalid"] and "association_evidence" in result:
            next(iter(result["association_evidence"].values()))["quote"] = "虚构引用"
        return httpx.Response(200, json={"message": {"content": json.dumps(result, ensure_ascii=False)}})
    pipeline.llm.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        first = pipeline.run(["2025-05-10"])
        assert first["llm_verification_fallbacks"] == 1 and first["accepted_events"] == 0
        assert not list((settings.paths.cache_dir / "days").glob("*.json"))
        state["invalid"] = False
        second = pipeline.run(["2025-05-10"])
        assert second["llm_verification_fallbacks"] == 0 and second["accepted_events"] == 1
        assert pipeline.llm.calls == 4  # extraction + link + bad review + repaired review
    finally:
        pipeline.close()


@pytest.mark.parametrize("association", ["uncertain", "different_activity"])
def test_location_difference_requires_confirmed_activity_scope(association):
    docs, observations, candidate = group(region="东海")
    observations[0].facts.update(region=["南海"], direction=["南海"])

    def review(request):
        result = normal_review(request)
        result["association"] = association
        return result

    event = VerificationAgent(VerificationSettings(), llm=ModelReply(review)).run([candidate], observations, docs)[0]
    assert event.classification == "untrusted" and not event.facts
    assert event.verdicts["direction"].status == event.verdicts["region"].status == "insufficient"
    assert not any(r.startswith("fact_conflict:") for r in event.verification_reasons)
    assert event.semantic_review["claims"]  # All differing values remain reviewable.


@pytest.mark.parametrize("status", ["supported", "conflict"])
def test_task_relationship_is_decided_by_model_with_complete_evidence(status):
    docs, observations, candidate = group()
    docs[0].text = docs[0].text.replace("战备巡逻", "两栖作战训练")
    docs[1].text = docs[1].text.replace("战备巡逻", "远海训练")
    extractor = ExtractionAgent(Settings(), None)
    observations[:2] = [extractor.text(d) for d in docs]
    observations[-1].facts["task"] = ["远海训练"]
    for row in observations[-1].metadata["raw_rows"]:
        row["任务"] = "远海训练"

    def review(request):
        result = normal_review(request)
        task = next(f for f in result["fields"] if f["field"] == "task")
        task.update(status=status, value="远海训练" if status == "supported" else None,
                    reason="模型完整审查宽泛训练类别与具体科目的关系")
        return result

    event = VerificationAgent(VerificationSettings(), llm=ModelReply(review)).run([candidate], observations, docs)[0]
    assert event.verdicts["task"].status == status
    assert event.classification == ("trusted" if status == "supported" else "untrusted")
    task_claims = [c for c in event.semantic_review["claims"] if c["field"] == "task"]
    assert {c["value"] for c in task_claims} == {"两栖作战训练", "远海训练"}


def test_candidate_model_receives_original_radar_rows():
    docs, observations, _ = group()
    model = ModelReply(normal_review)
    OneToOneAlignmentAgent(Settings().alignment, model).run(observations, docs)
    candidate = json.loads(model.requests[0][1])["radar_candidates"][0]
    assert candidate["radar_rows"][0]["values"]["活动区域"] == "东海中部"
    assert candidate["radar_rows"][0]["row_numbers"] == [2, 3]


def test_model_citation_spacing_is_restored_without_changing_words():
    from fusion_agent.semantics import anchored_quote
    text = "中国1艘054A型护卫舰在东海从11时37分至13时02分活动。"
    _, obs = semantic_text(text, {"country": [{"value": "中 国", "quote": "中国1艘054A型护卫舰"}],
                                  "platform": [{"value": "054 A", "quote": "054A 型护卫舰"}],
                                  "start": [{"value": "2025-05-01T11:37:00", "quote": "从 11 时 37 分"}]}, strict=True)
    assert obs.facts["country"] == ["中国"] and obs.facts["platform"] == ["054A"]
    assert obs.facts["start"] == ["2025-05-01T11:37:00"]
    assert obs.quotes["start"] == ["从11时37分"]
    assert obs.metadata["citation_whitespace_normalizations"]
    assert anchored_quote("中国舰艇在南海", text) is None
    assert anchored_quote("在东海，活动", "在东海。活动") is None


def test_model_number_cannot_be_used_as_individual_without_identity_cue():
    text = "韩国AW-159反潜直升机在阿拉伯海活动。"
    with pytest.raises(ValueError, match="型号/舰级应放platform"):
        semantic_text(text, {"individual": [{"value": "AW-159", "quote": text}]}, strict=True)
    explicit = "尾号AW-159的飞机在阿拉伯海活动。"
    _, obs = semantic_text(explicit, {"individual": [{"value": "AW-159", "quote": explicit}]}, strict=True)
    assert obs.metadata["individual_ids"]


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_real_client_retries_rejected_citations_and_never_reuses_bad_cache(tmp_path, repair_succeeds):
    from fusion_agent.llm import LLMClient
    settings = LLMSettings(ollama={"model": "fixture"}, retries=1, max_calls=2)
    client = LLMClient(settings, tmp_path)
    client.http.close()
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        quote = "真实原文" if repair_succeeds and len(requests) == 2 else "不存在的引用"
        return httpx.Response(200, json={"message": {"content": json.dumps({"quote": quote}, ensure_ascii=False)}})
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    schema = {"type": "object", "properties": {"quote": {"type": "string"}}, "required": ["quote"]}
    def validate(data):
        if data["quote"] != "真实原文":
            raise ValueError("引用未锚定原文")
        return data
    try:
        if repair_succeeds:
            assert client.validated_json("只引用原文", "真实原文", schema, validate)["quote"] == "真实原文"
        else:
            with pytest.raises(ValueError, match="引用未锚定"):
                client.validated_json("只引用原文", "真实原文", schema, validate)
        assert client.calls == 2 and client.validation_repairs == 1
        assert "引用未锚定原文" in requests[1]["messages"][-1]["content"]
        assert not client.response_path("只引用原文", "真实原文", schema).exists()
        assert len(list((tmp_path / "llm_rejections").glob("*.json"))) == (1 if repair_succeeds else 2)
    finally:
        client.close()


@pytest.mark.parametrize("counts", [(1, 1), (1, 2)])
def test_keyed_review_retains_all_evidence_and_cannot_hide_numeric_conflict(counts):
    def keyed_review(request):
        result = normal_review(request)
        result["fields"] = {d["field"]: {k: v for k, v in d.items() if k not in {"field", "claim_ids"}}
                            for d in result["fields"]}
        result["fields"]["quantity"].update(status="supported", value=1)
        result["association_evidence"] = {e["observation_id"]: {k: v for k, v in e.items() if k != "observation_id"}
                                            for e in result["association_evidence"]}
        result["excluded_claims"] = {}
        return result
    event, _, _ = verify(keyed_review, counts=counts)
    assert event.classification == ("trusted" if counts == (1, 1) else "untrusted")
    assert event.verdicts["quantity"].status == ("supported" if counts == (1, 1) else "conflict")
    audit = event.semantic_review
    assert sorted(cid for f in audit["response"]["fields"] for cid in f["claim_ids"]) == sorted(c["claim_id"] for c in audit["claims"])
    assert "claim_ids" not in audit["raw_response"]["fields"]["quantity"]


def test_review_context_uses_separate_cache_and_restores_after_failure(tmp_path):
    client = LLMClient(LLMSettings(ollama={"model": "fixture", "num_ctx": 8192, "review_num_ctx": 16384}), tmp_path)
    original = client.response_path("system", "source", {})
    try:
        with pytest.raises(ValueError):
            with client.review_context():
                assert client.settings.ollama.num_ctx == 16384
                assert client.response_path("system", "source", {}) != original
                raise ValueError("simulate failed review")
        assert client.settings.ollama.num_ctx == 8192
        assert client.response_path("system", "source", {}) == original
    finally:
        client.close()


def test_country_coverage_requires_model_assignment_and_allows_media_exclusion():
    text = "美国媒体称，意大利歼-15T在东海活动。"
    doc = Document(id="n", date="2025-05-01", source="open_source", path="n.txt", text=text)
    reply = {"activities": [{"activity_id": "one", "platform_kind": "air", "country": [],
                             "platform": [{"value": "歼-15T", "quote": "意大利歼-15T"}]}],
             "excluded_mentions": [{"quote": "美国媒体", "role": "background", "reason": "报道来源，不是活动主体"}]}
    settings = Settings()
    settings.llm.fail_on_error = True
    with pytest.raises(ValueError, match="llm_country_mentions_unreviewed:意大利"):
        ExtractionAgent(settings, ModelReply(reply)).text(doc)
    reply["activities"][0]["country"] = [{"value": "意大利", "quote": "意大利歼-15T"}]
    obs = ExtractionAgent(settings, ModelReply(reply)).text(doc)
    assert obs.facts["country"] == ["意大利"]
    assert obs.metadata["country_mention_coverage"]["excluded"] == ["美国"]


@pytest.mark.parametrize("region", ["菲律宾海", "印度洋"])
def test_country_coverage_respects_model_assigned_geographic_role(region):
    text = f"俄罗斯守护级护卫舰在{region}活动。"
    _, obs = semantic_text(text, {"country": [{"value": "俄罗斯", "quote": text}],
                                  "region": [{"value": region, "quote": text}]}, strict=True)
    assert obs.facts["country"] == ["俄罗斯"]
    assert not obs.metadata["country_mention_coverage"]["unresolved"]


@pytest.mark.parametrize("role", ["activity_subject", "uncertain"])
def test_missing_country_is_resolved_only_by_additional_model_role_decision(tmp_path, role):
    text = "美国媒体称，意大利歼-15T在东海活动。"
    doc = Document(id="n", date="2025-05-01", source="open_source", path="n.txt", text=text)
    settings = Settings(llm=LLMSettings(ollama={"model": "fixture"}, fail_on_error=True))
    client = LLMClient(settings.llm, tmp_path)
    client.http.close()
    def handler(request):
        payload = json.loads(request.content)
        if "activities" in payload["format"]["properties"]:
            activity = {f: [] for f in ("country", "platform", "region", "quantity", "task", "start", "end", "individual")}
            activity.update(activity_id="one", platform_kind="air", platform=[{"value": "歼-15T", "quote": text}])
            reply = {"activities": [activity]}
        else:
            reply = {"美国": {"role": "reporting_source", "quote": text, "activity_ids": [], "reason": "报道媒体的国家"},
                     "意大利": {"role": role, "quote": text, "activity_ids": ["one"] if role == "activity_subject" else [], "reason": "明确主体或保留角色不确定性"}}
        return httpx.Response(200, json={"message": {"content": json.dumps(reply, ensure_ascii=False)}})
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        obs = ExtractionAgent(settings, client).text(doc)
        assert client.calls == 2
        assert obs.facts.get("country") == (["意大利"] if role == "activity_subject" else [])
        assert obs.metadata["llm_extraction"]["activities"][0]["country"] == []
        assert obs.metadata["country_role_review"]["意大利"]["role"] == role
        assert ("country_role_uncertain" in obs.warnings) == (role == "uncertain")
    finally:
        client.close()


@pytest.mark.parametrize("phrase", ["围绕任务区域保持机动", "围绕任务海域保持机动", "在指定任务区域保持机动"])
def test_maneuver_alias_validates_selected_task_without_adding_unselected_task(phrase):
    text = f"中国054A型护卫舰在东海{phrase}。"
    country = [{"value": "中国", "quote": text}]
    _, selected = semantic_text(text, {"country": country, "task": [{"value": "区域机动", "quote": text}]}, strict=True)
    assert selected.facts["task"] == ["区域机动"]
    _, omitted = semantic_text(text, {"country": country}, strict=True)
    assert not omitted.facts.get("task")


@pytest.mark.parametrize("role", ["activity_fact", "uncertain"])
@pytest.mark.parametrize("refuses_text_correction", [False, True])
def test_excluded_counterclaim_requires_independent_model_role_check(tmp_path, role, refuses_text_correction):
    docs, observations, candidate = group()
    docs[1].text = docs[1].text.replace("1艘", "3艘")
    observations[1].facts["quantity"] = [3]
    observations[1].quotes["quantity"] = [docs[1].text]
    client = LLMClient(LLMSettings(ollama={"model": "fixture"}, retries=1, fail_on_error=True), tmp_path)
    client.http.close()
    final_calls = []
    def handler(request):
        payload = json.loads(request.content)
        data = json.JSONDecoder().raw_decode(payload["messages"][-1]["content"])[0]
        if "fields" not in payload["format"]["properties"]:
            assert "candidate_reviews" not in data
            assert {c["value"] for c in data["claims"]} == {3}
            reply = {c["claim_id"]: {"role": role, "quote": docs[1].text,
                                     "reason": "正文明确声明当前活动的数量，不能凭其他来源排除"} for c in data["claims"]}
        else:
            final_calls.append(payload)
            reply = normal_review(data)
            qty = next(f for f in reply["fields"] if f["field"] == "quantity")
            if len(final_calls) <= (2 if refuses_text_correction else 1):
                claim = next(c for c in data["claims"] if c["field"] == "quantity" and c["source"] == "opinion")
                qty["claim_ids"].remove(claim["claim_id"])
                qty.update(status="supported", value=1)
                reply["excluded_claims"] = [{"claim_id": claim["claim_id"], "role": "background",
                                             "reason": "其他两个来源写1，舆论写3可能误报"}]
            else:
                qty.update(status="conflict", value=None)
                if refuses_text_correction:
                    claim = next(c for c in data["claims"] if c["field"] == "quantity" and c["source"] == "opinion")
                    assert claim["claim_id"] not in payload["format"]["properties"]["excluded_claims"]["properties"]
                    assert data["independent_role_review"][claim["claim_id"]]["role"] == role
            reply["fields"] = {f["field"]: {k: v for k, v in f.items() if k not in {"field", "claim_ids"}}
                               for f in reply["fields"]}
            reply["excluded_claims"] = {c["claim_id"]: {k: v for k, v in c.items() if k != "claim_id"}
                                         for c in reply.get("excluded_claims", [])}
            reply["association_evidence"] = {e["observation_id"]: {k: v for k, v in e.items() if k != "observation_id"}
                                             for e in reply["association_evidence"]}
        return httpx.Response(200, json={"message": {"content": json.dumps(reply, ensure_ascii=False)}})
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        event = VerificationAgent(VerificationSettings(), llm=client).run([candidate], observations, docs)[0]
        assert event.classification == "untrusted" and event.verdicts["quantity"].status == "conflict"
        assert event.semantic_review["response"]["excluded_claims"] == []
        assert client.calls == (4 if refuses_text_correction else 3) and client.validation_repairs == 1
        assert "禁止当作背景排除" in final_calls[1]["messages"][-1]["content"]
    finally:
        client.close()


def test_independent_role_review_allows_confirmed_commentary_exclusion(tmp_path):
    from fusion_agent.agents.semantic_verification import SemanticVerifier
    client = LLMClient(LLMSettings(ollama={"model": "fixture"}), tmp_path)
    client.http.close()
    text = "评论认为该型舰具备反潜能力。"
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        "message": {"content": json.dumps({"C001": {"role": "commentary", "quote": text,
                                                     "reason": "评述平台能力，未声明当前活动任务"}}, ensure_ascii=False)}})))
    try:
        record = {"claim_id": "C001", "observation_id": "p", "source": "opinion", "field": "task",
                  "value": "反潜", "quotes": [text]}
        result = SemanticVerifier(client)._review_exclusions([SimpleNamespace(claim_id="C001")], {"C001": record}, {"p": text})
        assert result["C001"]["role"] == "commentary" and client.calls == 1
    finally:
        client.close()


def test_actor_country_exclusion_uses_source_local_field_definition_and_retains_counterfact(tmp_path):
    from fusion_agent.agents.semantic_verification import SemanticVerifier, ExclusionRoleError
    client = LLMClient(LLMSettings(ollama={"model": "fixture"}), tmp_path)
    client.http.close()
    text = "意大利歼-15T舰载战斗机在济州岛西部海域开展起降训练。"
    def handler(request):
        payload = json.loads(request.content)
        data = json.loads(payload["messages"][-1]["content"])
        assert set(data["sources"]) == {"n"} and "candidate_reviews" not in data
        assert "主体" in data["field_definitions"]["country"]
        assert "background" not in payload["format"]["properties"]["C001"]["properties"]["role"]["enum"]
        return httpx.Response(200, json={"message": {"content": json.dumps({"C001": {
            "role": "activity_fact", "quote": text, "reason": "国家修饰执行当前任务的平台，是主体国籍事实"}}, ensure_ascii=False)}})
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        record = {"claim_id": "C001", "observation_id": "n", "source": "open_source", "field": "country",
                  "value": "意大利", "quotes": [text]}
        with pytest.raises(ExclusionRoleError, match="禁止当作背景排除"):
            SemanticVerifier(client)._review_exclusions([SimpleNamespace(claim_id="C001")], {"C001": record}, {"n": text})
    finally:
        client.close()


def test_shared_model_activity_keeps_joint_platforms_without_formation_count():
    from fusion_agent.models import Observation
    docs, observations, candidate = group()
    news, opinion, first_radar = observations
    docs[0].text = "2025年5月1日，中国054A型护卫舰与056A型护卫舰共2艘在东海中部从10时00分至12时00分执行战备巡逻。"
    news.facts["platform"] = ["054A", "056A"]
    news.facts["quantity"] = [2]
    news.quotes["platform"] = [docs[0].text]
    news.quotes["quantity"] = [docs[0].text]
    news.metadata["platform_units"] = [news.model_copy(deep=True).model_dump(mode="json")]
    second_radar = first_radar.model_copy(deep=True)
    second_radar.id = "r2"
    second_radar.facts["platform"] = ["056A"]
    for row in second_radar.metadata["raw_rows"]:
        row["机舰名"] = "056A型护卫舰"
    observations.append(second_radar)
    candidate.observation_ids.append(second_radar.id)
    candidate.assignment = {"news_id": news.id}
    def review(request):
        result = normal_review(request)
        for field in result["fields"]:
            if field["field"] == "platform": field.update(status="supported", value="054A")
            if field["field"] == "quantity": field.update(status="insufficient", value=None)
        return result
    event = VerificationAgent(VerificationSettings(), llm=ModelReply(review)).run([candidate], observations, docs)[0]
    assert event.classification == "trusted"
    assert event.verdicts["platform"].value == "054A；056A"
    assert event.verdicts["platform"].support_sources == ["open_source", "radar"]
    assert event.platform_verdicts["056A"]["platform"].status == "supported"
    assert event.verdicts["quantity"].status == "insufficient"
    assert Observation.model_validate(news.metadata["platform_units"][0]).facts["quantity"] == [2]
