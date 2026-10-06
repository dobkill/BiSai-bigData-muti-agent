import pytest

from fusion_agent.agents.verification import VerificationAgent
from fusion_agent.config import VerificationSettings
from fusion_agent.models import Candidate, DayResult, Observation
from fusion_agent.knowledge import build_graph


def observations():
    facts = {"country": ["中国"], "platform": ["054A"], "direction": ["南海"],
             "region": ["礼乐滩海域"], "quantity": [1], "task": ["战备巡逻"],
             "start": ["2025-05-10T10:00:00"], "end": ["2025-05-10T12:00:00"]}
    return [Observation(id=s, date="2025-05-10", source=s, path=s, facts=facts)
            for s in ("open_source", "opinion", "radar")]


def verify(obs):
    c = Candidate(id="event", date="2025-05-10", observation_ids=[o.id for o in obs])
    return VerificationAgent(VerificationSettings()).run([c], obs)[0]


def test_three_agree_accept():
    event = verify(observations())
    assert event.accepted
    assert event.facts["quantity"] == 1
    assert set(event.verdicts["platform"].support_sources) == {"open_source", "opinion", "radar"}


@pytest.mark.parametrize("source", ["open_source", "opinion", "radar"])
@pytest.mark.parametrize("field,value", [("quantity", 2), ("task", "反潜巡逻"), ("direction", "东海"),
                                         ("country", "日本"), ("platform", "055"),
                                         ("start", "2025-05-10T11:00:00")])
def test_no_majority_override(source, field, value):
    obs = observations()
    next(o for o in obs if o.source == source).facts[field] = [value]
    event = verify(obs)
    assert not event.accepted
    assert "fact_conflict:" + field in event.reasons


def test_radar_internal_conflicts_preserved():
    obs = observations()
    obs[2].facts["quantity"] = [1, 2]
    assert not verify(obs).accepted


def test_third_source_unknown_can_be_supported_by_two():
    obs = observations()
    del obs[2].facts["quantity"]
    event = verify(obs)
    assert event.accepted
    assert event.facts["quantity"] == 1


@pytest.mark.parametrize("missing", ["open_source", "opinion", "radar"])
def test_missing_any_source_type_is_untrusted_even_when_two_agree(missing):
    event = verify([o for o in observations() if o.source != missing])
    assert not event.accepted and event.classification == "untrusted"
    assert "three_sources_not_linked" in event.verification_reasons
    assert event.verdicts["platform"].status == "supported"


def test_two_documents_of_one_source_type_are_untrusted():
    obs = observations()[:2]
    obs[1].source = "open_source"
    event = verify(obs)
    assert not event.accepted and event.classification == "untrusted"


def test_other_direction_can_be_trusted_without_entering_core_report():
    obs = observations()
    for o in obs:
        o.facts["direction"] = ["其他"]
        o.facts["region"] = ["阿拉伯海"]
    event = verify(obs)
    assert event.classification == "trusted" and not event.accepted
    assert event.verification_reasons == []
    assert event.reasons == ["outside_core_or_unknown_direction"]
    graph = build_graph([DayResult(date=event.date, observations=obs, candidates=[], events=[event], trace=[])])
    nodes = [n for n in graph["nodes"] if n["type"] == "event"]
    assert len(nodes) == 1 and not nodes[0]["report_accepted"]


def test_short_radar_track_is_unknown_support_but_can_still_contradict():
    obs = observations()
    obs[2].warnings = ["insufficient_radar_points"]
    event = verify(obs)
    assert event.classification == "trusted" and event.accepted
    assert set(event.verdicts["platform"].support_sources) == {"open_source", "opinion"}
    assert verify([obs[0], obs[2]]).classification == "untrusted"
    obs[2].facts["quantity"] = [2]
    assert verify(obs).classification == "untrusted"


@pytest.mark.parametrize("source", ["open_source", "opinion", "radar"])
def test_linked_third_source_can_omit_a_key_fact(source):
    obs = observations()
    next(o for o in obs if o.source == source).facts.pop("platform")
    event = verify(obs)
    assert event.classification == "trusted" and event.accepted
    assert set(event.verdicts["platform"].support_sources) == {o.source for o in obs} - {source}


def test_present_but_unlinked_third_source_does_not_make_event_trusted():
    obs = observations()
    obs[2].facts = {}
    event = verify(obs)
    assert event.classification == "untrusted" and not event.accepted
    assert "three_source_association_incomplete" in event.verification_reasons
    assert event.verdicts["platform"].status == "supported"


def test_all_three_sources_with_only_a_date_cannot_confirm_the_event():
    obs = observations()
    for o in obs:
        o.facts.pop("start")
        o.facts.pop("end")
    event = verify(obs)
    assert event.classification == "untrusted"
    assert "association_not_confirmed:candidate" in event.verification_reasons


def test_only_one_source_value_never_written():
    obs = observations()
    del obs[0].facts["quantity"]
    del obs[1].facts["quantity"]
    event = verify(obs)
    assert event.accepted
    assert "quantity" not in event.facts


def test_different_subregions_same_direction_still_conflict():
    obs = observations()
    obs[1].facts["region"] = ["黄岩岛海域"]
    assert not verify(obs).accepted


def test_specific_region_requires_two_specific_sources():
    obs = observations()
    obs[0].facts["region"] = ["南海"]
    obs[1].facts["region"] = ["南海"]
    assert verify(obs).facts["region"] == "南海"


def test_radar_track_anomaly_rejects_even_if_facts_agree():
    obs = observations()
    obs[2].warnings = ["radar_implausible_track_jump"]
    assert not verify(obs).accepted


def test_platform_kind_conflict_not_hidden_by_model_code():
    obs = observations()
    obs[0].metadata["platform_kinds"] = ["air"]
    obs[2].metadata["platform_kinds"] = ["ship"]
    assert "platform_kind_conflict" in verify(obs).reasons

