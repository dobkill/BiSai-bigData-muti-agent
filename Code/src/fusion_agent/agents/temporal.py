from collections import Counter, defaultdict
from datetime import date, timedelta

from ..models import DayResult


class TemporalAgent:
    name = "temporal"

    def __init__(self, lookback_days: int):
        self.lookback_days = lookback_days

    def run(self, target: str, days: list[DayResult]) -> dict:
        end = date.fromisoformat(target)
        start = end - timedelta(days=self.lookback_days - 1)
        # 严格只访问节点及过去数据，禁止未来信息泄漏。
        window = [d for d in days if start.isoformat() <= d.date <= target]
        events = [e for d in window for e in d.events if e.accepted]
        count = Counter(e.facts["direction"] for e in events)
        tracks = defaultdict(list)
        for event in events:
            key = (event.facts.get("country"), event.facts.get("platform"))
            tracks[key].append({"date": event.date, "direction": event.facts["direction"], "event_id": event.id})
        recurring = []
        for (country, platform), records in sorted(tracks.items()):
            observed_dates = sorted({r["date"] for r in records})
            if len(observed_dates) > 1:
                recurring.append({"country": country, "platform": platform, "observed_dates": observed_dates,
                                  "directions": sorted({r["direction"] for r in records}),
                                  "event_ids": [r["event_id"] for r in records],
                                  "interpretation": "同国别同型号多日出现；不等同于同一具体平台持续部署"})
        available = {d.date for d in window}
        expected = [(start + timedelta(days=i)).isoformat() for i in range(self.lookback_days)]
        return {"window_start": start.isoformat(), "window_end": target, "available_days": sorted(available),
                "missing_days": [d for d in expected if d not in available], "direction_counts": dict(count),
                "daily_counts": {d.date: sum(e.accepted for e in d.events) for d in window},
                "recurring_platform_types": recurring, "verified_event_count": len(events)}

