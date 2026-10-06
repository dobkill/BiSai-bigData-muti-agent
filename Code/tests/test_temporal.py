from fusion_agent.agents.temporal import TemporalAgent
from fusion_agent.models import DayResult, Event


def day(date):
    e = Event(id=date, date=date, accepted=True, evidence_ids=[],
              facts={"direction": "南海", "country": "中国", "platform": "054A"})
    return DayResult(date=date, events=[e], observations=[], candidates=[], trace=[])


def test_no_future_leakage_and_missing_days_not_zero():
    result = TemporalAgent(3).run("2025-05-10", [day("2025-05-08"), day("2025-05-10"), day("2025-05-11")])
    assert result["verified_event_count"] == 2
    assert result["missing_days"] == ["2025-05-09"]
    assert "2025-05-11" not in result["daily_counts"]
    assert result["recurring_platform_types"][0]["observed_dates"] == ["2025-05-08", "2025-05-10"]

