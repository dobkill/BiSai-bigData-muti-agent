"""Auditable literal assertions. No cross-source completion or military plausibility guesses."""
import re

from .data import compact, minutes
from .geography import location_relation, normalize_location

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
    pattern=r"(?:执行|组织|开展|展开|实施|进行|从事)(?:了)?\s*("+MISSION_PATTERN+r")"
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
    source["time_semantics"]=time_semantics(source)
    return source


def time_semantics(source):
    """An interval of reported activity does not assert its absolute beginning or end."""
    if source.get("time_semantics"):
        return source["time_semantics"]
    if source["kind"]=="radar":
        return {"interval":"observation_coverage","start_asserted":False,"end_asserted":False}
    text=narrative(source)
    clock=r"\d{1,2}\s*(?:时(?:\s*\d{1,2}\s*分)?|[:：]\s*\d{1,2})"
    return {"interval":"reported_activity",
            "start_asserted":bool(re.search(clock+r"\s*(?:开始|起(?:在|执行|组织|活动|，)|进入|驶入|飞入)",text)
                                  or re.search(r"(?:开始|进入|驶入|飞入)[^。]{0,12}"+clock,text)),
            "end_asserted":bool(re.search(clock+r"\s*(?:结束|离开|驶离|飞离|撤离)",text)
                                or re.search(r"(?:结束|离开|驶离|飞离|撤离)[^。]{0,12}"+clock,text))}


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
    # 远海 describes the training environment; 两栖/对海 etc. describe a subject.
    # They corroborate only the shared "训练" claim, never the unconfirmed specific subject.
    training_context=("远海训练" in mission_names and len(mission_names-{"远海训练"})<=1
                      and all(v.endswith("训练") for v in mission_names))
    if training_context and len(mission_names)>1:
        support=[sid for sid,values in missions.items() if values]
        status="supported" if len(source_types(support,assigned))>=2 else "insufficient"
        replace_fact(verification,"mission","训练",support,[],status,
                     f"原文明示任务：{missions}；远海是训练环境，其他训练名称是科目；只印证共同训练活动，不补作具体科目",
                     source_values=missions)
        for name in sorted(mission_names):
            ids=[sid for sid,values in missions.items() if name in values]
            verification["facts"].append({"platform_index":-1,"field":"mission_detail","claim":name,
                "supporting_sources":ids,"conflicting_sources":[],
                "status":"supported" if len(source_types(ids,assigned))>=2 else "insufficient",
                "reason":"环境/科目分别保留，仅按明确写出该细节的独立来源核验"})
    elif mission_names:
        best=max(mission_names,key=lambda v:(len(source_types([sid for sid,values in missions.items() if v in values],assigned)),v))
        support=[sid for sid,values in missions.items() if values==[best]]
        conflicts=[sid for sid,values in missions.items() if values and values!=[best]]
        status="conflict" if conflicts else "supported" if len(source_types(support,assigned))>=2 else "insufficient"
        reason=f"原文明示任务：{missions}；不同明确任务尚无包含/同时执行证据，不自动拼接或多数表决"
        replace_fact(verification,"mission",best,support,conflicts,status,reason,source_values=missions)
    else:
        replace_fact(verification,"mission","未明确任务",[],[],"insufficient","未找到正文任务断言；平台名称和机动描述不补作任务",source_values=missions)

    check_time_fact(verification,assigned)

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


def check_time_fact(verification,assigned):
    values={s["id"]:{"start":minutes(s["start_time"]) if s.get("start_time") else None,
                      "end":minutes(s["end_time"]) if s.get("end_time") else None,
                      **time_semantics(s)} for s in assigned}
    clock=lambda x:f"{x//60:02}:{x%60:02}"
    ranges={sid:(v["start"],v["end"]) for sid,v in values.items() if v["start"] is not None and v["end"] is not None}
    conflicts=set();boundaries={}
    for key,flag in [("start","start_asserted"),("end","end_asserted")]:
        exact={sid:v[key] for sid,v in values.items() if v[flag] and v[key] is not None}
        boundaries[key]={sid:clock(v) for sid,v in exact.items()}
        if len(set(exact.values()))>1:
            conflicts.update(exact)
        # Evidence before a positively asserted beginning or after an asserted end is a counter-value.
        for sid,bound in exact.items():
            for other,v in values.items():
                observed=v[key]
                if other!=sid and observed is not None and ((key=="start" and observed<bound) or (key=="end" and observed>bound)):
                    conflicts.update([sid,other])
    common=None
    if ranges:
        common=(max(x[0] for x in ranges.values()),min(x[1] for x in ranges.values()))
        if common[0]>common[1] or any(start>end for start,end in ranges.values()):
            conflicts.update(ranges)
    support=[];claim="活动时段未充分印证"
    if common and common[0]<=common[1]:
        support=list(ranges)
        claim=f"共同观测/报道时段 {clock(common[0])}—{clock(common[1])}（非完整活动起止断言）"
        # A lone endpoint confirms a point, not the entire interval following or preceding it.
        if len(source_types(support,assigned))<2:
            points={x for v in values.values() for x in [v["start"],v["end"]] if x is not None}
            confirmed=[(x,[sid for sid,v in values.items() if x in [v["start"],v["end"]]]) for x in sorted(points)]
            confirmed=[(x,ids) for x,ids in confirmed if len(source_types(ids,assigned))>=2]
            if confirmed:
                claim="共同支持的活动时刻 "+"、".join(clock(x) for x,ids in confirmed)
                support=sorted({sid for x,ids in confirmed for sid in ids})
    support=[sid for sid in support if sid not in conflicts]
    status="conflict" if conflicts else "supported" if len(source_types(support,assigned))>=2 else "insufficient"
    source_values={sid:{**v,"start":clock(v["start"]) if v["start"] is not None else None,
                        "end":clock(v["end"]) if v["end"] is not None else None} for sid,v in values.items()}
    reason=f"分别核验报道区间、观测覆盖与明示开始/结束；区间包含不构成起点冲突，首末雷达样本不等于完整活动边界；原始时段：{source_values}"
    replace_fact(verification,"time",claim,support,sorted(conflicts),status,reason,
                 source_values=source_values,asserted_boundaries=boundaries)


def check_location_fact(verification,assigned):
    explicit={s["id"]:s["location"] for s in assigned if s["location"].get("region")}
    original=next((dict(f) for f in verification["facts"] if f["field"]=="location" and f["platform_index"]==-1),{})
    regions={loc["region"] for loc in explicit.values()}
    # A concrete place with two exact assertions takes precedence over a merely broader statement.
    confirmed=[r for r in regions if len(source_types([sid for sid,l in explicit.items() if l["region"]==r],assigned))>=2]
    if confirmed:
        claim=max(confirmed,key=lambda r:(normalize_location(r)["precision"]!="大范围",len(r),r))
    elif regions:
        directions={l["direction"] for l in explicit.values()}
        specific={l["region"] for l in explicit.values() if l["precision"]!="大范围"}
        if len(directions)==1 and next(iter(directions)) in {"东海","南海","黄海"} and len(specific)<=1:
            claim=next(iter(directions))
        else:
            claim=next(iter(explicit.values()))["region"]
    else:
        replace_fact(verification,"location","未明确区域",[],[],"insufficient","没有明示区域；未验证坐标方向不补作地点断言")
        return
    target=normalize_location(claim)
    support=[];conflicts=[]
    for sid,loc in explicit.items():
        relation=location_relation(target,loc)
        if relation=="same" or relation=="contains" and target["precision"]=="大范围":
            support.append(sid)
        elif relation=="contains":
            pass  # A broader statement is compatible but does not assert the detailed location.
        else:
            conflicts.append(sid)
    # Widening cannot hide two differently named concrete regions, even within one core direction.
    specifics={l["region"] for l in explicit.values() if l["precision"]!="大范围"}
    if target["precision"]=="大范围" and len(specifics)>1:
        conflicts=[sid for sid,l in explicit.items() if l["precision"]!="大范围"]
        support=[sid for sid in support if sid not in conflicts]
    status="conflict" if conflicts else "supported" if len(source_types(support,assigned))>=2 else "insufficient"
    replace_fact(verification,"location",claim,support,conflicts,status,
                 f"复核明示地点和PDF包含关系：{ {sid:l['region'] for sid,l in explicit.items()} }；同方向不等于同一具体区域，未验证坐标不作为反证",
                 source_values=explicit)
    if status=="supported" and target["precision"]=="大范围" and len(specifics)==1:
        detail=next(iter(specifics));detail_sources=[sid for sid,l in explicit.items() if l["region"]==detail]
        if len(source_types(detail_sources,assigned))<2:
            verification["facts"].append({"platform_index":-1,"field":"location_detail","claim":detail,
                "supporting_sources":detail_sources,"conflicting_sources":[],"status":"insufficient",
                "reason":"具体区域仅一类来源明示；结论使用共同获支持的方向，细节保留待核实","model_proposal":original})
