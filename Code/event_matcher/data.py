import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path

import openpyxl

from .geography import coordinate_hint, normalize_location


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(text):
    return re.sub(r"[^A-Z0-9\u4e00-\u9fff]", "", (text or "").upper())


def platform_tokens(name: str):
    name = re.sub(r"\s+", "", name.upper())
    tokens = re.findall(r"[A-Z]{1,6}-?\d{1,4}[A-Z]*(?:/[A-Z])?|[歼轰运苏图直]-?\d+[A-Z]*|(?<!\d)\d{3}[A-Z]?(?!\d)", name)
    return {compact(t) for t in tokens}


def platform_similarity(a: str, b: str):
    ta, tb = platform_tokens(a), platform_tokens(b)
    if ta and tb:
        return 1.0 if ta & tb else 0.0
    ca, cb = compact(a), compact(b)
    for stop in ["海军", "空军", "导弹", "多用途", "反潜", "战斗机", "巡逻机", "护卫舰", "驱逐舰", "航空母舰", "常规潜艇", "型", "级"]:
        ca, cb = ca.replace(stop, ""), cb.replace(stop, "")
    if not ca or not cb:
        return 0.0
    if min(len(ca), len(cb)) >= 2 and (ca in cb or cb in ca):
        return 0.95
    ga = {ca[i:i+2] for i in range(len(ca)-1)}
    gb = {cb[i:i+2] for i in range(len(cb)-1)}
    return len(ga & gb) / max(1, len(ga | gb))


COUNTRY_ALIASES = {"中国海军": "中国", "中国空军": "中国", "俄军": "俄罗斯", "俄罗斯空天军": "俄罗斯",
                   "美军": "美国", "美海军": "美国", "美国海军": "美国", "美国空军": "美国",
                   "德军": "德国", "德国空军": "德国", "澳军": "澳大利亚", "澳海军": "澳大利亚",
                   "台军": "台湾", "台海军": "台湾", "台湾海军": "台湾", "英国空军": "英国"}


def country(value):
    if value is None:
        return None
    if compact(str(value)).casefold() in {"", "null", "none", "nan", "unknown", "未说明", "未明确", "未知", "不明", "不详", "未识别", "缺失"}:
        return None
    mapped = COUNTRY_ALIASES.get(value,value)
    names = [c for c in ["中国","韩国","日本","美国","俄罗斯","意大利","英国","澳大利亚","新西兰","德国","法国","菲律宾","越南","马来西亚","新加坡","印度","加拿大","台湾"] if c in mapped]
    return names[0] if len(names)==1 else mapped


def minutes(value):
    if not value:
        return None
    m = re.search(r"(\d{1,2}):(\d{2})(?::\d{2})?(?:\s*)$", value)
    if not m or int(m[1]) > 23 or int(m[2]) > 59:
        raise ValueError(f"invalid time {value}")
    return int(m[1]) * 60 + int(m[2])


def time_overlap(a, b):
    values = [a.get("start_time"), a.get("end_time"), b.get("start_time"), b.get("end_time")]
    if not all(values):
        return None
    a0, a1, b0, b1 = [minutes(v) for v in values]
    if a1 < a0:
        a1 += 1440
    if b1 < b0:
        b1 += 1440
    if min(a1, b1) < max(a0, b0):
        return 0.0
    return (min(a1, b1)-max(a0, b0)) / max(1, min(a1-a0, b1-b0))


def load_texts(day: Path):
    result = []
    for kind, dirname, prefix in [("news", "开源军情", "N"), ("opinion", "网上舆论", "O")]:
        for i, path in enumerate(sorted((day / dirname).glob("*.txt")), 1):
            raw = path.read_text(encoding="utf-8")
            title = re.search(r"标题[：:]\s*(.+)", raw)
            result.append({"id": f"{prefix}{i:03}", "kind": kind, "path": str(path.resolve()),
                           "filename": path.name, "title": title[1] if title else path.stem,
                           "text": raw, "sha256": sha256(path)})
    if not result:
        raise ValueError(f"no input texts in {day}")
    return result


def haversine(lon1, lat1, lon2, lat2):
    x1, y1, x2, y2 = map(math.radians, (lon1, lat1, lon2, lat2))
    a = math.sin((y2-y1)/2)**2 + math.cos(y1)*math.cos(y2)*math.sin((x2-x1)/2)**2
    return 6371 * 2 * math.asin(min(1, math.sqrt(a)))


def load_radar(day: Path):
    path = next(day.glob("*.xlsx"))
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    activities = []
    raw_count = 0
    for sheet in book:
        all_rows = list(sheet.iter_rows(values_only=True))
        headers = list(all_rows[0])
        batches = defaultdict(list)
        for rownum, values in enumerate(all_rows[1:], 2):
            if not any(v is not None for v in values):
                continue
            row = dict(zip(headers, values))
            batches[row["批号"]].append({"row": rownum, "values": row})
            raw_count += 1
        for batch, items in sorted(batches.items()):
            air = sheet.title == "飞机"
            clusters = [items]
            if air:
                # Stable origin position distinguishes mixed spatial strands. All remain linked by batch.
                clusters = []
                anchors = []
                for item in sorted(items, key=lambda x: (x["values"]["开始经度"], x["values"]["开始纬度"], x["row"])):
                    r = item["values"]
                    origin = r["开始经度"], r["开始纬度"]
                    cluster = next((i for i, anchor in enumerate(anchors) if haversine(*anchor, *origin) <= 100), None)
                    if cluster is None:
                        anchors.append(origin)
                        clusters.append([])
                        cluster = len(clusters)-1
                    clusters[cluster].append(item)
            for cluster_index, records in enumerate(clusters):
                uid = f"R{len(activities)+1:03}"
                vals = [r["values"] for r in records]
                def distinct(k):
                    return sorted({v[k] for v in vals}, key=str)
                def representative(k):
                    v = distinct(k)
                    return v[0] if len(v) == 1 else None
                name = representative("目标类型" if air else "机舰名") or "/".join(distinct("目标类型" if air else "机舰名"))
                countries = distinct("国家地区")
                quantities = distinct("目标数量" if air else "数量")
                missions = distinct("任务")
                time_key = "位置时间" if air else "发现时间"
                start = min(distinct(time_key)) if air else min(distinct("首点时间"))
                end = max(distinct(time_key)) if air else max(distinct("结束时间"))
                locations = [] if air else [normalize_location(v) for v in distinct("活动区域")]
                anomalies = []
                warnings = []
                if len(clusters) > 1:
                    warnings.append("同批号存在不同起始位置簇；未提供唯一目标身份，分支须分别关联，不能自动合并或当作本事件反证")
                for key in ["国家地区", "任务", "目标数量" if air else "数量"] + ([] if air else ["活动区域"]):
                    if len(distinct(key)) > 1:
                        anomalies.append(f"批号内部{key}存在不同值")
                hint = None
                coordinates = None
                if air:
                    lons, lats = sorted(v["经度"] for v in vals), sorted(v["纬度"] for v in vals)
                    coordinates = {"longitude_range": [lons[0], lons[-1]], "latitude_range": [lats[0], lats[-1]],
                                   "median": [lons[len(lons)//2], lats[len(lats)//2]],
                                   "origins": sorted({(v["开始经度"], v["开始纬度"]) for v in vals})}
                    hint = coordinate_hint(*coordinates["median"])
                activities.append({"id": uid, "kind": "radar", "sheet": sheet.title, "batch": batch,
                    "cluster": cluster_index, "sibling_ids": [], "path": str(path.resolve()), "headers": headers,
                    "platforms": [{"name": name, "country": countries[0] if len(countries)==1 else None,
                                   "quantity": quantities[0] if len(quantities)==1 else None,
                                   "quote": f"{sheet.title}工作表，批号{batch}", "quantity_quote": "目标数量/数量原始列"}],
                    "country_values": countries, "quantity_values": quantities, "mission_values": missions,
                    "mission": {"value": missions[0] if len(missions)==1 else None, "quote": "任务原始列"},
                    "location": locations[0] if len(locations)==1 else normalize_location(None),
                    "location_values": locations, "coordinates": coordinates, "coordinate_hint": hint,
                    "start_time": start, "end_time": end, "movement": "未说明",
                    "time_semantics": {"interval": "observation_coverage", "start_asserted": False, "end_asserted": False},
                    "identity_scope": "batch_is_not_unique_target", "warnings": warnings,
                    "hull_values": [] if air else distinct("机弦号"), "anomalies": anomalies,
                    "records": sorted(records, key=lambda x:x["row"]), "row_count": len(records)})
    book.close()
    for a in activities:
        a["sibling_ids"] = [b["id"] for b in activities if (b["sheet"], b["batch"])==(a["sheet"], a["batch"]) and b["id"] != a["id"]]
    return activities, {"path": str(path.resolve()), "sha256": sha256(path), "rows": raw_count,
                        "batches": len({(a["sheet"],a["batch"]) for a in activities}), "activities": len(activities)}


def public_view(source):
    keys = ["id", "kind", "title", "text", "platforms", "location", "location_values", "mission", "mission_values",
            "start_time", "end_time", "movement", "coordinates", "coordinate_hint", "anomalies", "batch",
            "sibling_ids", "hull_values", "country_values", "quantity_values", "row_count",
            "time_semantics", "identity_scope", "warnings"]
    return {k: source[k] for k in keys if k in source}
