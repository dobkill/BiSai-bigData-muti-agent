"""Protocol fixtures only: these replies are never used for dataset reruns."""
import json
from fusion_agent.semantics import association_quote_options


def fixture_reply(payload):
    result = _fixture_reply(payload)
    schema = payload.get("format", {})
    if not isinstance(schema, dict):
        schema = {}
    response_format = payload.get("response_format", {})
    schema = response_format.get("schema", response_format.get("json_schema", {}).get("schema", schema))
    props = schema.get("properties", {})
    if "decisions" in result and props.get("decisions", {}).get("type") == "object":
        for name, identifier in (("decisions", "opinion_id"), ("radar_decisions", "observation_id")):
            result[name] = {d[identifier]: {k: v for k, v in d.items() if k != identifier} for d in result[name]}
    if "fields" in result and props.get("fields", {}).get("type") == "object":
        result["fields"] = {d["field"]: {k: v for k, v in d.items() if k not in {"field", "claim_ids"}} for d in result["fields"]}
        result["association_evidence"] = {e["observation_id"]: {k: v for k, v in e.items() if k != "observation_id"}
                                            for e in result["association_evidence"]}
        result["excluded_claims"] = {}
    return result


def _fixture_reply(payload):
    user = payload["messages"][-1]["content"]
    if user.startswith("事件默认日期："):
        text = user.split("原文：\n", 1)[1]
        return {"activities": [{"activity_id": "one", "platform_kind": "ship",
                                "country": [{"value": "中国", "quote": text}],
                                "platform": [{"value": "054A", "quote": text}],
                                "region": [{"value": "南海", "quote": text}],
                                "quantity": [{"value": 1, "quote": text}],
                                "task": [{"value": "战备巡逻", "quote": text}],
                                "start": [], "end": [], "individual": []}]}
    request = json.loads(user)
    if "opinion_candidates" in request:
        return {"decisions": [{"opinion_id": p["opinion_id"], "decision": "same_event",
                               "news_quote": association_quote_options(request["news_text"])[0],
                               "opinion_quote": association_quote_options(p["text"])[0],
                               "reason": "原文主体、区域、活动上下文一致"} for p in request["opinion_candidates"]],
                "radar_decisions": [{"observation_id": r["observation_id"], "decision": "same_event",
                                     "news_quote": association_quote_options(request["news_text"])[0], "row_numbers": r["row_numbers"],
                                     "reason": "对应主体及活动海域"} for r in request["radar_candidates"]]}
    fields = []
    for field in ("country", "platform", "direction", "region", "quantity", "task", "start", "end"):
        claims = [c for c in request["claims"] if c["field"] == field]
        values = list(dict.fromkeys(c["value"] for c in claims))
        sources = {c["source"] for c in claims}
        status = "absent" if not claims else "conflict" if len(values) > 1 else "supported" if len(sources) >= 2 else "insufficient"
        fields.append({"field": field, "status": status, "value": values[0] if status == "supported" else None,
                       "claim_ids": [c["claim_id"] for c in claims], "reason": "协议测试中的显式证据对照"})
    return {"association": "same_activity", "reason": "协议测试中的三源活动归属",
            "association_evidence": [{"observation_id": s["observation_id"], "quote": association_quote_options(s["text"])[0],
                                      "row_numbers": sorted({n for row in s["radar_rows"] for n in row["row_numbers"]})}
                                     for s in request["sources"]], "fields": fields}
