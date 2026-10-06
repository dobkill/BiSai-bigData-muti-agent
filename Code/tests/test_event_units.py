from fusion_agent.agents.event_units import EventUnitAgent
from fusion_agent.agents.extraction import ExtractionAgent
from fusion_agent.config import Settings
from fusion_agent.domain import platform_catalog
from fusion_agent.models import Document


def units(text):
    doc = Document(id="doc", date="2025-05-10", source="open_source", path="doc.txt", text=text)
    extraction = ExtractionAgent(Settings(), None)
    extraction.platform_aliases = platform_catalog(["金刚级导弹驱逐舰"])
    return EventUnitAgent().run([doc], [extraction.text(doc)])


def test_joint_platform_counts_are_assigned_only_to_named_target():
    result = units("正文：2025年5月10日，英国1架P-8A反潜巡逻机和2架RC-135W电子侦察机在南海从10时00分至12时00分联合活动。")
    assert len(result) == 2
    assert {o.facts["platform"][0]: o.facts["quantity"] for o in result} == {"P-8A": [1], "RC-135W": [2]}
    assert all(o.metadata["parent_observation_id"] == "doc" for o in result)
    assert all("multiple_platforms_unresolved" not in o.warnings for o in result)


def test_formation_total_is_not_assigned_to_each_platform():
    result = units("正文：2025年5月10日，中国P-8A和RC-135W组成3架飞机的编队在南海活动。")
    assert len(result) == 2
    assert all("quantity" not in o.facts for o in result)
    assert all(o.metadata["unassigned_fields"]["quantity"] == [3] for o in result)


def test_ship_class_and_hull_number_are_not_two_platforms():
    result = units("正文：2025年5月10日，日本金刚级导弹驱逐舰（DDG-173）在东海活动。")
    assert len(result) == 1
    assert result[0].facts["platform"] == ["金刚级"]
    assert result[0].metadata["individual_ids"] == ["DDG173"]


def test_separate_sentences_keep_country_direction_and_time_local():
    result = units("正文：2025年5月10日，中国歼-16战斗机在南海从10时00分至12时00分活动。美国F-35B战斗机在东海从14时00分至16时00分活动。")
    assert len(result) == 2
    by_platform = {o.facts["platform"][0]: o for o in result}
    assert by_platform["歼-16"].facts["country"] == ["中国"]
    assert by_platform["F-35B"].facts["direction"] == ["东海"]
    assert by_platform["F-35B"].facts["start"] == ["2025-05-10T14:00:00"]
    assert all("invalid_time_interval" not in o.warnings for o in result)


def test_publication_time_does_not_become_activity_time():
    result = units("发文时间：2025年5月10日10时00分至12时00分\n正文：2025年5月10日，中国歼-16战斗机在南海活动。")
    assert "start" not in result[0].facts and "end" not in result[0].facts
    assert result[0].metadata["time_precision"] == "date_only"
