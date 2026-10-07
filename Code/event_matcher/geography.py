"""Contest taxonomy. Direction membership is not a legal/geometric boundary."""
import re

REGIONS = {
    "东海": "东海 东海北部 东海中部 东海南部 钓鱼岛海域 赤尾屿海域 上海外海 长江口外海 舟山外海 宁波外海 温州外海 东海油气田海域 冲绳海槽 苏岩礁附近海域".split(),
    "南海": "南海 南海北部 南海中部 南海南部 北部湾 琼州海峡 海南岛东南海域 珠江口外海 东沙群岛海域 西沙群岛海域 中沙群岛海域 南沙群岛海域 黄岩岛海域 美济礁附近海域 仁爱礁附近海域 万安滩海域 礼乐滩海域 曾母暗沙海域 永暑礁附近海域 渚碧礁附近海域 北康暗沙海域 南康暗沙海域 纳土纳群岛北部海域".split(),
    "黄海": "黄海 北黄海 黄海中部 南黄海 青岛外海 成山头外海 海州湾外海 连云港外海 济州岛西部海域".split(),
    "其他": "台湾海峡 巴士海峡 巴林塘海峡 宫古水道 对马海峡".split(),
}
REGION_DIRECTION = {r: d for d, regions in REGIONS.items() for r in regions}
ALIASES = {r.removesuffix("海域"): r for r in REGION_DIRECTION if r.endswith("海域")}
ALIASES.update({"南海海域": "南海", "东海海域": "东海", "黄海海域": "黄海", "钓鱼岛附近海域": "钓鱼岛海域", "赤尾屿附近海域": "赤尾屿海域"})
TRANSIT = set(REGIONS["其他"])


def normalize_location(raw: str | None) -> dict:
    if not raw:
        return {"raw": None, "region": None, "direction": None, "precision": "缺失", "mapping": "missing"}
    text = re.sub(r"\s+", "", raw)
    # Models sometimes include an adjacent predicate in an otherwise literal location span.
    # Keep raw unchanged, but do not turn '阿拉伯海围绕任务海域' into a new place name.
    text=re.split(r"围绕|执行|组织|开展|持续|展开|进行|按计划|完成|沿既定|实施|保持",text,maxsplit=1)[0]
    text=re.sub(r"^(?:在|于)","",text)
    matches = [(len(k), v) for k, v in {**{r: r for r in REGION_DIRECTION}, **ALIASES}.items() if k in text]
    if matches:
        region = max(matches)[1]
        direction = REGION_DIRECTION[region]
        return {"raw": raw, "region": region, "direction": direction,
                "precision": "大范围" if region in {"东海", "南海", "黄海"} else "具体区域",
                "mapping": "PDF第2页活动区域表", "transit": region in TRANSIT}
    # Unlisted named regions are Other; missing is deliberately not Other.
    region=re.sub(r"(?:相关海域|海空域|海域|方向)$","",text) or text
    return {"raw": raw, "region": region, "direction": "其他", "precision": "原文区域",
            "mapping": "PDF未列入三大核心海域的区域归其他", "transit": False}


def location_relation(a: dict, b: dict) -> str:
    if not a.get("region") or not b.get("region"):
        return "missing"
    if a["region"] == b["region"]:
        return "same"
    if a["direction"] == b["direction"] and a["direction"] != "其他":
        if a["precision"] == "大范围" or b["precision"] == "大范围":
            return "contains"
        return "same_direction_only"
    if a["direction"] != b["direction"]:
        return "different_direction"
    return "different_named_regions"


def coordinate_hint(lon: float, lat: float) -> dict:
    # Conservative broad boxes, solely for recall. Never count as verified source support.
    candidates = []
    if 120 <= lon <= 130 and 24 <= lat <= 33:
        candidates.append("东海")
    if 118 <= lon <= 127 and 32 <= lat <= 40:
        candidates.append("黄海")
    if 105 <= lon <= 122 and -1 <= lat <= 23:
        candidates.append("南海")
    return {"candidate_directions": candidates, "verified": False,
            "method": "宽泛矩形，仅辅助召回；PDF没有坐标边界，不能作为独立地点印证或反证"}
