"""Publication invariants, not an accuracy score or an automatic GPT reviewer."""
from .ollama_client import dump


def audit_quality(groups,texts,radars,output):
    byid={s["id"]:s for s in texts+radars}
    errors=[];facts_checked=0
    for group in groups:
        assigned={group["news_id"],*group["radar_ids"]}
        if group["opinion_id"]:
            assigned.add(group["opinion_id"])
        if not assigned<=set(byid):
            errors.append({"group":group["id"],"problem":"引用未知材料ID"})
            continue
        facts=group["verification"]["facts"]
        for fact in facts:
            support=set(fact["supporting_sources"]);counter=set(fact["conflicting_sources"])
            if not (support|counter)<=assigned or support & counter:
                errors.append({"group":group["id"],"field":fact["field"],"problem":"事实引用越出本组或支持反对重叠"})
                continue
            if fact["status"]=="supported":
                facts_checked+=1
                if len({byid[sid]["kind"] for sid in support})<2 or counter:
                    errors.append({"group":group["id"],"field":fact["field"],"problem":"已确认事实不足两类支持或存在未消解反证"})
        if group["status"]=="可信事件":
            kinds={byid[sid]["kind"] for sid in assigned}
            if kinds!={"news","opinion","radar"} or group["reasons"] or any(f["status"]=="conflict" for f in facts):
                errors.append({"group":group["id"],"problem":"可信分类与缺源/反证/判定原因不一致"})
        elif not group["reasons"]:
            errors.append({"group":group["id"],"problem":"不可信没有可复核依据"})
    result={"passed":not errors,"groups_checked":len(groups),"supported_facts_checked":facts_checked,"errors":errors,
            "scope":"检查来源独立性、引用范围、分类自洽；不判断军事事实真伪，不替代GPT逐组阅读与官方真值评估"}
    dump(output/"运行记录"/"质量检查.json",result)
    if errors:
        raise RuntimeError("输出质量约束失败："+str(errors))
    return result
