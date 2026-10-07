"""Auditable literal assertions. No cross-source completion or military plausibility guesses."""
import re

from .data import compact, minutes

# Task names, not platform roles; suffixes such as 任务/任务行动 are grammatical wrappers.
MISSION_NAMES = (
    "情报监视侦察", "海上侦察巡逻", "海上监视侦察", "舰载机起降训练", "远程轰炸训练",
    "联合搜救训练", "两栖作战训练", "对海突击训练", "实弹射击训练", "电子对抗训练",
    "编队警戒部署", "预警指挥", "反潜巡逻", "电子侦察", "战备警巡", "战备巡逻",
    "空中警戒", "联合巡航", "联合巡逻", "海上护航", "海上巡逻", "远海训练", "例行部署", "自由航行",
)
MISSION_PATTERN = "|".join(re.escape(v) for v in sorted(MISSION_NAMES,key=len,reverse=True))


def normalize_mission(value):
    if not value:
        return None
    matches={m.group() for m in re.finditer(MISSION_PATTERN,compact(value))}
    return next(iter(matches)) if len(matches)==1 else None


def narrative(source):
    raw=source.get("text","")
    title=re.search(r"标题[：:]\s*(.+)",raw)
    body=re.search(r"正文[：:]\s*([\s\S]+)",raw)
    first=(body[1].split("。",1)[0] if body else raw)
    return (title[1]+"。" if title else "")+first


def literal_missions(source):
    if source["kind"]=="radar":
        return {v:v for v in (normalize_mission(x) for x in source.get("mission_values",[source["mission"].get("value")])) if v}
    text=narrative(source)
    # A platform called 反潜巡逻机 is not itself an assertion of a 反潜巡逻 mission.
    pattern=r"(?:执行|组织|开展|实施|进行|从事)(?:了)?\s*("+MISSION_PATTERN+r")"
    return {m[1]:m.group() for m in re.finditer(pattern,text)}


def literal_movement(source):
    if source["kind"]=="radar":
        return "未说明"
    text=narrative(source)
    entering=bool(re.search(r"进入|驶入|飞入",text))
    leaving=bool(re.search(r"离开|驶离|飞离|撤离",text))
    if entering and leaving:
        return "进入及离开"
    return "进入" if entering else "离开" if leaving else "活动"


def repair_source_assertions(source):
    """Repair only source-internal literal errors and retain the model's original proposal."""
    proposed={}
    missions=literal_missions(source)
    old=normalize_mission(source["mission"].get("value"))
    if len(missions)<=1 and old != next(iter(missions),None):
        proposed["mission"]=dict(source["mission"])
        name=next(iter(missions),None)
        source["mission"]={"value":name,"quote":missions[name] if name else ""}
        source["extraction_repairs"].append("按正文任务动词复核；平台用途名称及泛称活动不构成明确任务")
    if len(missions)>1:
        source["mission_values"]=sorted(missions)
        source["anomalies"]=["文本内部出现不同明确任务，须核实是否同一活动"]
    movement=literal_movement(source)
    if source["movement"]!=movement:
        proposed["movement"]={"value":source["movement"],"quote":source.get("movement_quote")}
        source["movement"]=movement
        source["extraction_repairs"].append("按原文明示进出动词校正行动关系，保持机动不能推断进入")
    if proposed:
        source["extraction_model_proposal"]=proposed
    return source


def source_types(ids,assigned):
    kinds={s["id"]:s["kind"] for s in assigned}
    return {kinds[sid] for sid in ids}


def replace_fact(verification,field,claim,support,conflicts,status,reason,**extra):
    fact=next((f for f in verification["facts"] if f["field"]==field and f["platform_index"]==-1),None)
    if fact is None:
        fact={"platform_index":-1,"field":field}
        verification["facts"].append(fact)
    proposal=dict(fact)
    fact.update({"claim":claim,"supporting_sources":sorted(set(support)),"conflicting_sources":sorted(set(conflicts)),
                 "status":status,"reason":reason,"model_proposal":proposal,"checked_by":"literal_assertion_matrix",**extra})
    return fact


def check_common_facts(verification,assigned):
    missions={s["id"]:sorted(literal_missions(s)) for s in assigned}
    mission_names={v for values in missions.values() for v in values}
    if mission_names:
        best=max(mission_names,key=lambda v:(len(source_types([sid for sid,values in missions.items() if v in values],assigned)),v))
        support=[sid for sid,values in missions.items() if values==[best]]
        conflicts=[sid for sid,values in missions.items() if values and values!=[best]]
        status="conflict" if conflicts else "supported" if len(source_types(support,assigned))>=2 else "insufficient"
        reason=f"原文明示任务：{missions}；不同明确任务尚无包含/同时执行证据，不自动拼接或多数表决"
        replace_fact(verification,"mission",best,support,conflicts,status,reason,source_values=missions)
    else:
        replace_fact(verification,"mission","未明确任务",[],[],"insufficient","未找到正文任务断言；平台名称和机动描述不补作任务",source_values=missions)

    components={}
    for key in ["start_time","end_time"]:
        observed={s["id"]:minutes(s[key]) for s in assigned if s.get(key)}
        if not observed:
            components[key]={"value":None,"support":[],"conflicts":[],"status":"insufficient"}
            continue
        best=max(set(observed.values()),key=lambda v:(len(source_types([sid for sid,x in observed.items() if x==v],assigned)),-v))
        support=[sid for sid,x in observed.items() if x==best]
        conflicts=[sid for sid,x in observed.items() if x!=best]
        status="conflict" if conflicts else "supported" if len(source_types(support,assigned))>=2 else "insufficient"
        components[key]={"value":f"{best//60:02}:{best%60:02}","support":support,"conflicts":conflicts,
                         "status":status,"source_values":{sid:f"{x//60:02}:{x%60:02}" for sid,x in observed.items()}}
    support=sorted({sid for c in components.values() for sid in c["support"]})
    conflicts=sorted({sid for c in components.values() for sid in c["conflicts"]})
    # A source can agree on one endpoint and contradict another. Keep per-endpoint evidence.
    support=[sid for sid in support if sid not in conflicts]
    status="conflict" if conflicts else "supported" if all(c["status"]=="supported" for c in components.values()) else "insufficient"
    claim="起点 "+str(components["start_time"]["value"] or "未说明")+"，终点 "+str(components["end_time"]["value"] or "未说明")
    reason="逐端点检查活动时刻；单端点不被丢弃，缺失不反对。不同端点未获分阶段/观测覆盖差异的证据解释时保留待核实冲突"
    replace_fact(verification,"time",claim,support,conflicts,status,reason,components=components)

    movements={s["id"]:literal_movement(s) for s in assigned}
    enters=[sid for sid,v in movements.items() if v=="进入"]
    leaves=[sid for sid,v in movements.items() if v=="离开"]
    active=[sid for sid,v in movements.items() if v!="未说明"]
    if enters and leaves:
        replace_fact(verification,"movement","进出关系待核实",enters,leaves,"conflict",f"同活动进出断言不同：{movements}",source_values=movements)
    else:
        detailed=enters if len(source_types(enters,assigned))>=2 else leaves if len(source_types(leaves,assigned))>=2 else []
        claim="进入" if detailed==enters and detailed else "离开" if detailed else "活动"
        support=detailed or active
        status="supported" if len(source_types(support,assigned))>=2 else "insufficient"
        replace_fact(verification,"movement",claim,support,[],status,"活动可兼容进出过程，但活动描述不能支持具体进入/离开；明示进出细节保留在source_values",source_values=movements)
