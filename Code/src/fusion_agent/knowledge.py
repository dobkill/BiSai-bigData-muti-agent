"""保存已核验事件的知识图谱和按字段的证据边。"""
from .models import DayResult


def build_graph(days: list[DayResult]) -> dict:
    nodes = {}
    edges = []
    for day in days:
        observation_index = {o.id: o for o in day.observations}
        for event in day.events:
            trusted = event.classification == "trusted" if event.classification is not None else event.accepted
            if not trusted:
                continue
            eid = "event:" + event.id
            nodes[eid] = {"id": eid, "type": "event", "date": event.date, "facts": event.facts,
                          "classification": event.classification, "report_accepted": event.accepted}
            for kind in ("country", "platform", "direction", "region", "task"):
                if value := event.facts.get(kind):
                    nid = kind + ":" + str(value)
                    nodes[nid] = {"id": nid, "type": kind, "label": value}
                    edges.append({"source": eid, "target": nid, "relation": "has_" + kind})
            for oid in event.evidence_ids:
                obs = observation_index[oid]
                nid = "evidence:" + oid
                nodes[nid] = {"id": nid, "type": "evidence", "source_type": obs.source, "path": obs.path,
                              "sheet": obs.metadata.get("sheet"), "row_numbers": obs.metadata.get("row_numbers")}
                supported = [f for f,v in event.verdicts.items() if v.status == "supported" and oid in v.evidence_ids]
                edges.append({"source": nid, "target": eid, "relation": "supports", "fields": supported})
    return {"nodes": list(nodes.values()), "edges": edges}

