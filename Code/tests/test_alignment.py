import pytest

from fusion_agent.agents.alignment import AlignmentAgent
from fusion_agent.agents.verification import VerificationAgent
from fusion_agent.config import AlignmentSettings, VerificationSettings
from fusion_agent.models import Observation
from fusion_agent.relations import RelationMatcher


def obs(source="open_source", oid="n", start="10:00", end="12:00", **facts):
    data = {"country": ["中国"], "platform": ["054A"], "direction": ["南海"],
            "region": ["礼乐滩海域"], "task": ["战备巡逻"], "quantity": [1]}
    if start:
        data["start"] = ["2025-05-10T" + start + ":00"]
    if end:
        data["end"] = ["2025-05-10T" + end + ":00"]
    data.update(facts)
    return Observation(id=oid, date="2025-05-10", source=source, path=oid, facts=data,
                       metadata={"platform_kinds": ["ship"]})


@pytest.mark.parametrize("field,value", [("country", "日本"), ("platform", "055"), ("direction", "东海")])
def test_strong_conflicts_cannot_be_offset_by_other_agreements(field, value):
    relation = RelationMatcher(AlignmentSettings()).compare(obs(), obs("radar", "r", **{field: [value]}))
    assert relation.status == "conflicted"
    assert relation.fields[field] == "conflict"


def test_interval_containment_is_confirmable_without_equal_endpoints():
    relation = RelationMatcher(AlignmentSettings()).compare(obs(), obs("radar", "r", "10:30", "11:30"))
    assert relation.status == "confirmed"
    assert relation.time_relation == "overlap"


def test_several_hour_gap_is_candidate_and_not_confirmation():
    relation = RelationMatcher(AlignmentSettings()).compare(obs(), obs("radar", "r", "14:00", "15:00"))
    assert relation.status == "candidate"
    assert relation.gap_minutes == 120
    assert RelationMatcher(AlignmentSettings()).compare(obs(), obs("radar", "r", "18:00", "19:00")).status == "unrelated"


def test_hours_gap_is_uncertain_time_rather_than_explicit_fact_conflict():
    observations = [obs(), obs("opinion", "p"), obs("radar", "r", "14:00", "15:00")]
    candidate = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)[0]
    event = VerificationAgent(VerificationSettings()).run([candidate], observations)[0]
    assert candidate.relation_status == "candidate" and not event.accepted
    assert event.verdicts["start"].status == event.verdicts["end"].status == "insufficient"
    assert not any(reason.startswith("fact_conflict:") for reason in event.reasons)


def test_date_only_and_generic_identity_do_not_prove_same_event():
    matcher = RelationMatcher(AlignmentSettings())
    assert matcher.compare(obs(start=None, end=None), obs("radar", "r")).status == "candidate"
    assert matcher.compare(obs(platform=["generic:护卫舰"]), obs("radar", "r")).status == "candidate"


def test_missing_country_does_not_become_conflict():
    relation = RelationMatcher(AlignmentSettings()).compare(obs(country=[]), obs("radar", "r"))
    assert relation.fields["country"] == "unknown"
    assert relation.status == "confirmed"


def test_quantity_conflict_keeps_relation_and_blocks_fact_acceptance():
    observations = [obs(), obs("opinion", "p"), obs("radar", "r", quantity=[2])]
    candidates = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)
    assert len(candidates) == 1
    assert candidates[0].relation_status == "conflicted"
    event = VerificationAgent(VerificationSettings()).run(candidates, observations)[0]
    assert not event.accepted and "fact_conflict:quantity" in event.reasons


def test_alternatives_are_preserved_instead_of_selecting_first():
    observations = [obs(), obs("opinion", "p"), obs("radar", "r1"), obs("radar", "r2")]
    candidates = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)
    assert len(candidates) == 2
    assert all(c.relation_status == "candidate" for c in candidates)
    assert all("multiple_compatible_hypotheses" in c.warnings for c in candidates)
    assert all(set(c.observation_ids) >= {"n", "p"} for c in candidates)


def test_multiple_reports_can_share_one_confirmed_event():
    observations = [obs(), obs(oid="n2"), obs("opinion", "p"), obs("radar", "r")]
    candidates = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)
    assert len(candidates) == 1
    assert candidates[0].relation_status == "confirmed"
    assert set(candidates[0].observation_ids) == {"n", "n2", "p", "r"}


def test_one_source_date_only_can_be_confirmed_with_other_two_and_unique_identity():
    observations = [obs(start=None, end=None), obs("opinion", "p"), obs("radar", "r")]
    candidate = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)[0]
    assert candidate.relation_status == "confirmed"
    assert any(r.time_relation == "date_only" for r in candidate.relations)


@pytest.mark.parametrize("named_identity", [False, True])
def test_all_sources_date_only_are_not_confirmed(named_identity):
    observations = [obs(start=None, end=None), obs("opinion", "p", None, None), obs("radar", "r", None, None)]
    if named_identity:
        for observation in observations:
            observation.metadata["individual_ids"] = ["DDG173"]
    assert AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)[0].relation_status == "candidate"


def test_only_radar_has_time_even_same_individual_and_region_is_candidate():
    observations = [obs(start=None, end=None), obs("opinion", "p", None, None), obs("radar", "r")]
    for observation in observations:
        observation.metadata["individual_ids"] = ["DDG173"]
    candidate = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)[0]
    assert candidate.relation_status == "candidate"
    assert all(r.time_relation == "date_only" for r in candidate.relations)


def test_triangle_is_checked_not_only_two_radar_edges():
    observations = [obs(end="10:30"), obs("opinion", "p", "11:00", "11:30"), obs("radar", "r")]
    candidate = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)[0]
    assert candidate.relation_status == "candidate"
    assert len(candidate.relations) == 3


def test_overlap_allows_fact_verification_but_does_not_invent_boundaries():
    observations = [obs(), obs("opinion", "p", "10:15", "11:45"), obs("radar", "r", "10:30", "11:30")]
    candidates = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)
    event = VerificationAgent(VerificationSettings()).run(candidates, observations)[0]
    assert event.accepted
    assert "start" not in event.facts and "end" not in event.facts


def test_broad_direction_does_not_hide_different_specific_regions():
    relation = RelationMatcher(AlignmentSettings()).compare(obs(), obs("radar", "r", region=["黄岩岛海域"]))
    assert relation.status == "conflicted"
    assert relation.fields["region"] == "conflict"


def test_conflicting_hull_numbers_cannot_be_hidden_by_same_class():
    a, b = obs(), obs("radar", "r")
    a.metadata["individual_ids"] = ["DDG173"]
    b.metadata["individual_ids"] = ["DDG174"]
    relation = RelationMatcher(AlignmentSettings()).compare(a, b)
    assert relation.status == "conflicted"
    assert relation.fields["individual"] == "conflict"


def test_nearby_hours_alternative_does_not_override_clear_overlap():
    observations = [obs(), obs("opinion", "p"), obs("radar", "r"), obs("opinion", "p2", "14:00", "15:00")]
    candidates = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)
    confirmed = [c for c in candidates if c.relation_status == "confirmed"]
    assert len(confirmed) == 1 and "p" in confirmed[0].observation_ids
    assert any("p2" in c.observation_ids for c in candidates)


@pytest.mark.parametrize("opinion_already_grouped", [False, True])
@pytest.mark.parametrize("reverse_input", [False, True])
def test_text_pair_is_preserved_when_news_is_already_in_radar_conflict_group(opinion_already_grouped, reverse_input):
    context = {"direction": ["东海"], "region": ["苏岩礁附近海域"]}
    news = obs(**context)
    opinion = obs("opinion", "p", None, None, **context)
    radar = obs("radar", "r", direction=["黄海"], region=[])
    observations = [news, opinion, radar]
    if opinion_already_grouped:
        observations += [obs(oid="n2", start="20:00", end="22:00", **context),
                         obs("radar", "r2", "20:00", "22:00", **context)]
    if reverse_input:
        observations.reverse()
    matcher = RelationMatcher(AlignmentSettings())
    assert matcher.compare(news, radar).status == "conflicted"
    assert matcher.compare(news, opinion).status == "candidate"
    candidates = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)
    assert any(set(c.observation_ids) == {"n", "p"} and c.relation_status == "candidate" for c in candidates)
    assert not any(c.relation_status == "unmatched" and "p" in c.observation_ids for c in candidates)


def test_different_named_seas_are_not_merged_through_missing_context():
    from fusion_agent.agents.extraction import ExtractionAgent
    from fusion_agent.models import Document
    from fusion_agent.config import Settings

    extractor = ExtractionAgent(Settings(), None)
    texts = [
        ("n1", "open_source", "中国1艘903A型补给舰在成山头外海持续活动。"),
        ("n2", "open_source", "中国903A型补给舰在菲律宾海从14时44分至16时09分执行海上护航。"),
        ("p", "opinion", "中国1艘903A型补给舰在菲律宾海执行海上护航。"),
    ]
    observations = [extractor.text(Document(id=oid, date="2025-05-10", source=source,
                                          path=oid+".txt", text="正文：2025年5月10日，"+body))
                    for oid, source, body in texts]
    observations.append(obs("radar", "r", "14:44", "16:09", platform=["903A"],
                            country=["中国"], direction=[], region=[], task=["海上护航"]))
    candidates = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)
    assert not any({"n1", "n2"} <= set(c.observation_ids) for c in candidates)
    assert any({"n2", "p", "r"} == set(c.observation_ids) for c in candidates)


def test_unmatched_basis_explains_explicit_country_difference():
    observations = [obs(country=["意大利"], start=None, end=None),
                    obs("opinion", "p", None, None, country=["澳大利亚"])]
    candidates = AlignmentAgent(AlignmentSettings(mode="legacy")).run(observations)
    assert all(c.relation_status == "unmatched" for c in candidates)
    news = next(c for c in candidates if "n" in c.observation_ids)
    assert any("同型号舆论" in line and "国家不同（意大利 / 澳大利亚）" in line for line in news.basis)
