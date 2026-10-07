import pytest

from fusion_agent.agents.extraction import ExtractionAgent, quantities, text_times
from fusion_agent.config import Settings
from fusion_agent.domain import coordinate_direction, platforms, regions, tasks, platform_catalog, REGION_DIRECTION
from fusion_agent.models import Document, RadarBatch


def test_explicit_quantity_only_and_model_normalization():
    assert quantities("中国歼-16在南海活动") == []
    assert quantities("两架歼-16和3架直升机") == [2, 3]
    assert platforms("中国海军054A型护卫舰") == ["054A"]
    assert platforms("两架空警600") == ["空警-600"]


def test_region_longest_match_and_transit_classification():
    assert regions("在南海南沙群岛海域活动") == ["南海", "南沙群岛海域"]
    assert regions("在台湾海峡") == ["台湾海峡"]
    assert coordinate_direction(123, 31, 0.3) == "东海"
    assert coordinate_direction(123, 32, 0.3) is None


def test_philippine_sea_is_explicit_region_and_not_unknown():
    assert regions("中国903A型补给舰在菲律宾海持续活动") == ["菲律宾海"]
    assert REGION_DIRECTION["菲律宾海"] == "其他"
    agent = ExtractionAgent(Settings(), None)
    doc = Document(id="n", date="2025-05-10", source="open_source", path="n.txt",
                   text="标题：中国903A型补给舰在菲律宾海活动\n正文：2025年5月10日，中国903A型补给舰在菲律宾海活动。")
    observation = agent.text(doc)
    assert observation.facts["country"] == ["中国"]
    assert observation.facts["region"] == ["菲律宾海"]
    assert observation.facts["direction"] == ["其他"]


def test_radar_explicit_philippine_sea_is_preserved():
    row = {"国家地区": "中国", "机舰名": "903A型补给舰", "数量": 1, "任务": "海上护航",
           "活动区域": "菲律宾海", "首点时间": "2025.05.10 14:44:00", "结束时间": "2025.05.10 16:09:00"}
    batch = RadarBatch(id="r", date="2025-05-10", path="a.xlsx", sheet="舰船", batch="SHIP-1",
                       rows=[row, dict(row)], row_numbers=[2, 3])
    observation = ExtractionAgent(Settings(), None).radar(batch)
    assert observation.facts["region"] == ["菲律宾海"]
    assert observation.facts["direction"] == ["其他"]


def test_event_date_over_publication_date():
    start, end = text_times("2025年5月9日11时21分至13时46分活动", "2025-05-10")
    assert start == ["2025-05-09T11:21:00"]
    assert end == ["2025-05-09T13:46:00"]


def test_platform_description_is_not_task_and_nested_tasks():
    assert tasks("P-8A反潜巡逻机执行联合巡逻") == ["联合巡逻"]
    assert tasks("电子侦察船在海上活动") == []
    assert tasks("直升机执行联合搜救训练") == ["联合搜救训练"]
    assert tasks("预警机执行预警指挥") == ["预警指挥"]


def test_non_latin_model_names_and_dynamic_ship_catalog():
    assert platforms("俄罗斯苏-30SM多用途战斗机") == ["苏-30SM"]
    assert platforms("F/A-18E/F舰载战斗机") == ["F/A-18E/F"]
    catalog = platform_catalog(["世宗大王级导弹驱逐舰", "23型导弹护卫舰"])
    assert platforms("韩国世宗大王级驱逐舰", catalog) == ["世宗大王级"]
    assert platforms("英国23型护卫舰") == ["23"]
    assert platforms("罗纳德·里根号航空母舰", platform_catalog(["罗纳德·里根号航空母舰"])) == ["里根号"]


def test_radar_batch_preserves_internal_count_noise():
    rows = [{"国家地区": "中国", "机舰名": "054A型护卫舰", "数量": q, "任务": "战备巡逻",
             "活动区域": "南海", "首点时间": "2025.05.10 10:00:00", "结束时间": "2025.05.10 12:00:00"}
            for q in [1, 2]]
    batch = RadarBatch(id="r", date="2025-05-10", path="a.xlsx", sheet="舰船", batch="SHIP-1", rows=rows, row_numbers=[2,3])
    obs = ExtractionAgent(Settings(), None).radar(batch)
    assert obs.facts["quantity"] == [1, 2]
    assert obs.metadata["row_numbers"] == [2, 3]


class FakeLLM:
    def complete_json(self, *args):
        return {"activities": [{"activity_id": "one", "platform_kind": "ship",
                                "quantity": [{"value": 99, "quote": "根本不存在的引文"}],
                                "country": [{"value": "美国", "quote": "中国"}]}]}


def test_llm_hallucination_cannot_add_unsupported_fact():
    doc = Document(id="d", date="2025-05-10", source="open_source", path="d.txt",
                   text="标题：中国054A型护卫舰在南海活动\n正文：中国054A型护卫舰在南海活动。")
    obs = ExtractionAgent(Settings(), FakeLLM()).text(doc)
    assert not obs.facts.get("country")  # do not repair a bad model value with rule facts
    assert "quantity" not in obs.facts
    assert "llm_unanchored_claim:quantity" in obs.warnings


@pytest.mark.parametrize("text,quote,expected", [
    ("中国歼-15舰载战斗机在黄海活动", "舰载战斗机", ["歼-15"]),
    ("菲律宾FA-50PH轻型战斗机在南海活动", "轻型战斗机", ["FA-50PH"]),
    ("日本金刚级导弹驱逐舰在东海活动", "导弹驱逐舰", ["金刚级"]),
    ("中国一架战斗机在黄海活动", "战斗机", ["generic:战斗机"]),
    ("中国歼-15与歼-16战斗机在黄海活动", "战斗机", ["歼-15", "歼-16"]),
])
def test_llm_platform_descriptors_do_not_create_false_multi_platform(text, quote, expected):
    class DescriptorLLM:
        def complete_json(self, *args):
            return {"activities": [
                        {"activity_id": str(i), "platform_kind": "unknown",
                         "platform": [{"value": value.removeprefix("generic:"), "quote": text}]}
                        for i, value in enumerate(expected)]}

    doc = Document(id="d", date="2025-05-10", source="open_source", path="d.txt", text=text)
    agent = ExtractionAgent(Settings(), DescriptorLLM())
    agent.platform_aliases = platform_catalog(["金刚级导弹驱逐舰"])
    obs = agent.text(doc)
    assert obs.facts["platform"] == expected
    assert ("multiple_platforms_unresolved" in obs.warnings) == (len(expected) > 1)
    assert obs.metadata["extractor"] == "llm_semantic"
    assert len(obs.metadata["semantic_activities"]) == len(expected)

