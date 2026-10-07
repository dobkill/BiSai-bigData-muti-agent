import pytest

from fusion_agent.agents.alignment import AlignmentAgent
from fusion_agent.agents.event_units import EventUnitAgent
from fusion_agent.agents.extraction import ExtractionAgent
from fusion_agent.agents.verification import VerificationAgent
from fusion_agent.config import Settings
from fusion_agent.models import Document, RadarBatch


DAY = "2025-05-01"


def air_row(lon, lat, time="19:00:00", **extra):
    return {"国家地区": "中国", "目标类型": "歼-35舰载隐身战斗机", "目标数量": 2,
            "任务": "战备警巡", "经度": lon, "纬度": lat, "开始经度": lon, "开始纬度": lat,
            "位置时间": DAY + " " + time, **extra}


def ship_row(region="温州外海", start="17:35:00", end="19:00:00", **extra):
    return {"国家地区": "中国", "机舰名": "054A型护卫舰", "机弦号": "TYPE-054A", "数量": 2,
            "任务": "战备警巡", "活动区域": region, "首点时间": DAY + " " + start,
            "结束时间": DAY + " " + end, **extra}


def batch(rows, sheet="飞机"):
    return RadarBatch(id="radar", date=DAY, path="radar.xlsx", sheet=sheet, batch="BATCH-1",
                      rows=rows, row_numbers=list(range(2, len(rows) + 2)))


def extract(rows, sheet="飞机", docs=None):
    settings = Settings()
    result = ExtractionAgent(settings, None).run(docs or [], [batch(rows, sheet)])
    return EventUnitAgent().run(docs or [], result, keep_documents=True)


def test_anonymous_aircraft_positions_are_not_one_track():
    rows = [air_row(122.2, 27.7), air_row(114.3, 15.2)]
    units = extract(rows)
    assert len(units) == 2
    assert {tuple(o.facts["direction"]) for o in units} == {("东海",), ("南海",)}
    assert all(not o.metadata["track_identity_confirmed"] and set(o.warnings) <= {"insufficient_radar_points"} for o in units)
    assert sorted(n for o in units for n in o.metadata["row_numbers"]) == [2, 3]
    assert all(o.metadata["batch"] == "BATCH-1" and o.metadata["batch_row_numbers"] == [2, 3] for o in units)
    assert sorted(r["经度"] for o in units for r in o.metadata["raw_rows"]) == [114.3, 122.2]


def test_named_aircraft_impossible_movement_survives_activity_partition():
    units = extract([air_row(122.2, 27.7, 目标编号="PLANE-01"),
                     air_row(114.3, 15.2, 目标编号="PLANE-01")])
    assert len(units) == 2
    assert all(o.metadata["individual_ids"] == ["PLANE01"] for o in units)
    assert all(o.metadata["track_identity_confirmed"] for o in units)
    assert all("radar_implausible_track_jump" in o.warnings for o in units)
    assert all(o.metadata["track_check_row_numbers"] == [2, 3] for o in units)


def test_named_aircraft_can_move_between_directions_without_automatic_conflict():
    units = extract([air_row(123, 33, "17:00:00", 目标编号="PLANE-01"),
                     air_row(123, 31, "18:00:00", 目标编号="PLANE-01")])
    assert len(units) == 2 and all(set(o.warnings) <= {"insufficient_radar_points"} for o in units)


def test_aircraft_with_different_tail_numbers_do_not_create_jump():
    units = extract([air_row(122.2, 27.7, 尾号="101"), air_row(114.3, 15.2, 尾号="102")])
    assert len(units) == 2 and all(set(o.warnings) <= {"insufficient_radar_points"} for o in units)


def test_anonymous_same_class_ships_in_different_regions_are_separate_activities():
    units = extract([ship_row(), ship_row("万安滩海域")], "舰船")
    assert len(units) == 2
    assert {tuple(o.facts["direction"]) for o in units} == {("东海",), ("南海",)}
    assert all(o.metadata["individual_ids"] == [] for o in units)


def test_same_named_ship_overlapping_location_contradiction_is_retained():
    units = extract([ship_row(机弦号="FFG-01"), ship_row("万安滩海域", 机弦号="FFG-01")], "舰船")
    assert len(units) == 1
    assert units[0].metadata["individual_ids"] == ["FFG01"]
    assert units[0].facts["direction"] == ["东海", "南海"]


def test_same_named_ship_at_separate_times_is_not_one_activity():
    units = extract([ship_row(机弦号="FFG-01"),
                     ship_row("舟山外海", "20:00:00", "21:00:00", 机弦号="FFG-01")], "舰船")
    assert len(units) == 2
    assert all(o.metadata["individual_ids"] == ["FFG01"] for o in units)
    assert all(len(o.facts["start"]) == len(o.facts["end"]) == 1 for o in units)


@pytest.mark.parametrize("field,value", [("数量", 3), ("任务", "联合巡逻"), ("国家地区", "美国")])
def test_conflicting_attributes_are_not_used_to_split_the_same_activity(field, value):
    units = extract([ship_row(), ship_row(**{field: value})], "舰船")
    assert len(units) == 1
    target = {"数量": "quantity", "任务": "task", "国家地区": "country"}[field]
    assert len(units[0].facts[target]) == 2


@pytest.mark.parametrize("sheet", ["飞机", "舰船"])
def test_three_source_verification_uses_only_matching_activity(sheet):
    platform = "歼-35舰载隐身战斗机" if sheet == "飞机" else "054A型护卫舰"
    amount = "两架" if sheet == "飞机" else "两艘"
    docs = [Document(id="news", date=DAY, source="open_source", path="004.txt",
                     text=f"正文：2025年5月1日17时35分至19时00分，中国{platform}在温州外海执行战备警巡。"),
            Document(id="opinion", date=DAY, source="opinion", path="021.txt",
                     text=f"正文：2025年5月1日，中国{amount}{platform}在东海执行战备警巡。")]
    rows = ([air_row(122.2, 27.7, "17:49:10"), air_row(122.2, 27.7),
             air_row(114.3, 15.2, "17:35:00"), air_row(114.3, 15.2)] if sheet == "飞机" else
            [ship_row(), ship_row(), ship_row("万安滩海域"), ship_row("万安滩海域")])
    observations = extract(rows, sheet, docs)
    aligner = AlignmentAgent(Settings().alignment)
    candidates = aligner.run(observations, docs)
    event = VerificationAgent(Settings().verification).run(candidates, observations)[0]
    assert event.classification == "trusted" and event.accepted
    assert event.facts["direction"] == "东海" and event.facts["quantity"] == 2
    assert len(aligner.unassigned_observation_ids) == 1
    unused = next(o for o in observations if o.id in aligner.unassigned_observation_ids)
    assert unused.facts["direction"] == ["南海"]
    if sheet == "飞机":
        assert event.verdicts["start"].status == "insufficient"
        assert "start" not in event.facts  # south-sea 17:35 observation must not support this event


def test_different_named_ships_are_unlinked_instead_of_fact_conflicts():
    units = extract([ship_row(机弦号="FFG-01"), ship_row(机弦号="FFG-02")], "舰船")
    from fusion_agent.relations import RelationMatcher
    relation = RelationMatcher(Settings().alignment).compare(*units)
    assert relation.status == "unrelated" and relation.conflicts == []
    assert relation.fields["individual"] == "different"


def test_shared_ship_name_with_contradictory_hull_numbers_still_requires_review():
    from fusion_agent.models import Observation
    from fusion_agent.relations import RelationMatcher
    facts = {"country": ["中国"], "platform": ["054A"], "direction": ["东海"],
             "start": [DAY + "T17:35:00"], "end": [DAY + "T19:00:00"]}
    a = Observation(id="a", date=DAY, source="open_source", path="a", facts=facts,
                    metadata={"individual_ids": ["甲号", "FFG01"]})
    b = Observation(id="b", date=DAY, source="radar", path="b", facts=facts,
                    metadata={"individual_ids": ["甲号", "FFG02"]})
    relation = RelationMatcher(Settings().alignment).compare(a, b)
    assert relation.status == "conflicted" and "individual" in relation.conflicts


def test_shared_ship_identity_alias_does_not_hide_internal_hull_noise():
    units = extract([ship_row(机舰名="甲号护卫舰", 机弦号="FFG-01"),
                     ship_row(机舰名="甲号护卫舰", 机弦号="FFG-02")], "舰船")
    assert len(units) == 1
    assert "radar_individual_identity_conflict" in units[0].warnings
    assert set(units[0].metadata["individual_ids"]) == {"甲号", "FFG01", "FFG02"}
