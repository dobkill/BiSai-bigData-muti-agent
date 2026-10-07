"""Partition batch containers into activity evidence, without inventing target identity."""
from collections import defaultdict
from datetime import datetime, timedelta
import json
import re

from .config import Settings
from .domain import coordinate_direction, platforms, regions, REGIONS
from .models import RadarBatch
from .utils import parse_time, stable_id, unique


def normalized_identifier(value):
    return re.sub(r"[\s\-]", "", str(value or "")).upper()


def identifier_kind(value):
    return "number" if re.fullmatch(r"[A-Z]*\d+[A-Z]*", value) else "name"


def row_individual_ids(row, air, aliases=None):
    # Batch IDs, model codes and target counts identify containers/types, not individuals.
    columns = ("目标编号", "飞机编号", "机号", "尾号") if air else ("舷号", "机弦号")
    identifiers = []
    for column in columns:
        raw = row.get(column)
        token = normalized_identifier(raw)
        if token and not token.startswith("TYPE") and (air or re.search(r"\d", token)):
            identifiers.append(token)
    if not air:
        identifiers += [normalized_identifier(p) for p in platforms(str(row.get("机舰名") or ""), aliases)
                        if p.endswith("号")]
    return unique(identifiers)


def _window(row, air):
    start = parse_time(row.get("位置时间" if air else "首点时间"))
    end = start if air else parse_time(row.get("结束时间"))
    if start and end and end >= start:
        return datetime.fromisoformat(start), datetime.fromisoformat(end)
    return None


def _subject_context(row, air, settings, aliases, identity_keys):
    ids = row_individual_ids(row, air, aliases)
    lon, lat = row.get("经度"), row.get("纬度")
    direction = (coordinate_direction(lon, lat, settings.verification.geo_boundary_margin_degrees)
                 if air and isinstance(lon, (int, float)) and isinstance(lat, (int, float)) else None)
    if ids:
        # Shared explicit aliases link identity claims; disagreements remain in the unit.
        identity = identity_keys[ids[0]]
        return ("individual", identity, direction) if air else ("individual", identity)
    names = tuple(normalized_identifier(p) for p in platforms(
        str(row.get("目标类型" if air else "机舰名") or ""), aliases))
    if air:
        origin = (row.get("开始经度"), row.get("开始纬度"))
        origin = origin if all(isinstance(v, (int, float)) for v in origin) else None
        return ("activity", names, origin, direction)
    rs = regions(str(row.get("活动区域") or ""))
    specific = [r for r in rs if r not in REGIONS]
    return ("activity", names, tuple(specific or rs))


def split_radar_batch(batch: RadarBatch, settings: Settings, aliases=None):
    """Return disjoint row subsets and provenance; never split on quantity/task/country.

    Known individuals stay together during overlapping activities, so contradictory
    locations or attributes remain visible. Anonymous observations are grouped by
    activity context, without asserting that they are the same aircraft/ship.
    """
    air = batch.sheet == "飞机"
    parents = {}
    def find(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = find(parents[value])
        return parents[value]
    for row in batch.rows:
        ids = row_individual_ids(row, air, aliases)
        if ids:
            for value in ids:
                left, right = find(ids[0]), find(value)
                parents[max(left, right)] = min(left, right)
    identity_keys = {value: find(value) for value in parents}
    contexts = defaultdict(list)
    for number, row in zip(batch.row_numbers, batch.rows):
        contexts[_subject_context(row, air, settings, aliases, identity_keys)].append((number, row, _window(row, air)))
    groups = []
    slack = timedelta(hours=settings.alignment.candidate_gap_hours) if air else timedelta(0)
    for context, rows in contexts.items():
        dated = sorted((r for r in rows if r[2]), key=lambda r: (r[2][0], r[0]))
        undated = [r for r in rows if not r[2]]
        component, finish = [], None
        for entry in dated:
            start, end = entry[2]
            if component and start > finish + slack:
                groups.append((context, component))
                component, finish = [], None
            component.append(entry)
            finish = max(finish, end) if finish else end
        if component:
            groups.append((context, component))
        if undated:
            groups.append((context, undated))
    if len(groups) <= 1:
        return [(batch, {})]
    result = []
    for context, entries in groups:
        entries.sort(key=lambda r: r[0])
        starts = [r[2][0].isoformat() for r in entries if r[2]]
        key = json.dumps([context, min(starts) if starts else None], ensure_ascii=False)
        unit = batch.model_copy(update={"id": stable_id(batch.id + "|activity|" + key),
                                        "rows": [r[1] for r in entries], "row_numbers": [r[0] for r in entries]})
        result.append((unit, {"parent_radar_id": batch.id, "batch_row_numbers": batch.row_numbers,
                              "batch_activity_count": len(groups), "activity_context": context,
                              "activity_basis": "按明确目标编号或型号、活动位置及时间区分批号内活动；批号不作为单目标身份"}))
    return result
