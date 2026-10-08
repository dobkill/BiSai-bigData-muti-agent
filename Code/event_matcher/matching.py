import json
import sqlite3

import numpy as np

from .agents import align_source, quantity_numbers, verify_group
from .assignment import maximum_assignment
from .data import compact, country, minutes, platform_similarity, public_view, time_overlap
from .geography import location_relation, normalize_location
from .ollama_client import dump
from .literal_facts import check_common_facts, check_location_fact, literal_missions


def embedding_text(source):
    platforms = "; ".join(f"{p['country'] or '国籍未说明'} {p['name']} 数量{p['quantity'] if p['quantity'] is not None else '未说明'}" for p in source["platforms"])
    location = source.get("location", {})
    hint = source.get("coordinate_hint") or {}
    return (f"平台：{platforms}。活动区域：{location.get('region') or '未说明'}。"
            f"活动方向：{location.get('direction') or '未说明'}。"
            f"坐标辅助召回方向：{'、'.join(hint.get('candidate_directions', []))}。"
            f"活动时段：{source.get('start_time') or '未说明'} 至 {source.get('end_time') or '未说明'}。"
            f"任务：{source['mission'].get('value') or '未说明'}。行动：{source.get('movement', '未说明')}。")


def build_index(client, sources, output):
    texts = [embedding_text(s) for s in sources]
    indexdir = output / "运行记录" / "向量索引"
    # Backfill per-document caches from a previous validated batch cache to avoid re-embedding unchanged evidence.
    if (indexdir/"evidence.sqlite").exists() and (indexdir/"manifest.json").exists():
        import hashlib
        previous=json.loads((indexdir/"manifest.json").read_text(encoding="utf-8"))
        if previous["model"]==client.embedding_model:
            with sqlite3.connect(indexdir/"evidence.sqlite") as db:
                previous_texts=dict(db.execute("SELECT id,retrieval_text FROM sources"))
            previous_texts=[previous_texts[sid] for sid in previous["ids"]]
            key=hashlib.sha256(json.dumps([client.embedding_model,previous_texts],ensure_ascii=False).encode()).hexdigest()
            cached=client.cache/"embeddings"/(key+".json")
            if cached.exists():
                client.seed_embeddings(previous_texts,json.loads(cached.read_text(encoding="utf-8"))["vectors"])
    vectors = np.asarray(client.embeddings(texts), dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("embedding output is not a finite 2D matrix")
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    indexdir.mkdir(parents=True, exist_ok=True)
    np.save(indexdir / "vectors.npy", vectors)
    with sqlite3.connect(indexdir / "evidence.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, kind TEXT, region TEXT, direction TEXT, retrieval_text TEXT, evidence_json TEXT, embedding BLOB)")
        db.execute("DELETE FROM sources")
        db.executemany("INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(s["id"], s["kind"], s["location"].get("region"), s["location"].get("direction"), text,
              json.dumps(s, ensure_ascii=False), vector.tobytes()) for s,text,vector in zip(sources,texts,vectors)])
    manifest = {"model": client.embedding_model, "records": len(sources), "dimension": int(vectors.shape[1]),
                "metric": "normalized cosine; exact exhaustive search", "ids": [s["id"] for s in sources],
                "source": "Ollama /api/embed, cached", "storage": "SQLite证据和向量BLOB + vectors.npy"}
    dump(indexdir / "manifest.json", manifest)
    return vectors @ vectors.T, manifest


def recall(news, pool, similarities, positions, full_scores=False):
    scored = []
    for source in pool:
        ps = max(platform_similarity(a["name"], b["name"]) for a in news["platforms"] for b in source["platforms"])
        overlap = time_overlap(news, source)
        lr = location_relation(news["location"], source["location"])
        place_score = {"same": 1, "contains": .6, "same_direction_only": .2}.get(lr, 0)
        cs = any(a.get("country") and a.get("country") == b.get("country") for a in news["platforms"] for b in source["platforms"])
        cosine = float(similarities[positions[news["id"]], positions[source["id"]]])
        score = ps*3 + (overlap or 0)*1.2 + place_score*.8 + cs*.2 + cosine
        scored.append({"id": source["id"], "score": round(score, 5), "cosine": round(cosine, 5),
                       "platform_similarity": round(ps, 5), "time_overlap": overlap, "location_relation": lr})
    top_vector = sorted(scored, key=lambda s:(-s["cosine"], s["id"]))[:5]
    top_rules = sorted(scored, key=lambda s:(-s["score"], s["id"]))[:3]
    selected = {s["id"] for s in top_vector+top_rules} | {s["id"] for s in scored if s["platform_similarity"] >= .9}
    records = sorted((s for s in scored if s["id"] in selected), key=lambda s:(-s["score"],s["id"]))
    byid = {s["id"]: s for s in pool}
    return [byid[s["id"]] for s in records], scored if full_scores else records


def expanded_candidates(news,pool,scores,limit=12):
    """Search the complete indexed pool; prioritize compatible identities in a bounded review."""
    byid={s["id"]:s for s in pool}
    ranked=[r["id"] for r in sorted(scores,key=lambda r:(-r["score"],r["id"]))]
    compatible=[s["id"] for s in pool if any(platform_similarity(p["name"],q["name"])>=.45
                                           for p in news["platforms"] for q in s["platforms"])]
    selected=set(ranked[:limit])|set(compatible)
    # Unsent candidates remain in the index and raw export; this call has an explicit context budget.
    order=list(dict.fromkeys([*[sid for sid in ranked if sid in compatible],*compatible,*ranked]))
    return [byid[sid] for sid in order if sid in selected][:max(limit,3*len(news["platforms"]))]


def edge_weight(edge, heuristic):
    if not edge or edge["relation"] == "different" or edge["confidence"] < .55:
        return -1000.0
    factor = 1.0 if edge["relation"] == "same" else .45
    return edge["confidence"]*100*factor + min(heuristic, 6)


def assignment_competitors(weights,chosen,ids,owners,row,rival_ids):
    """Force a competing edge and recompute the entire optimum, not a local score gap."""
    baseline=sum(weights[i][j] for i,j in enumerate(chosen))
    result=[]
    for rival in sorted(set(rival_ids)):
        column=ids.index(rival)
        if weights[row][column]<=0:
            continue
        forced=[list(values) for values in weights]
        forced[row]=[-1000000.0]*len(forced[row])
        forced[row][column]=weights[row][column]
        alternative=maximum_assignment(forced)
        if alternative[row]!=column:
            raise RuntimeError("反事实分配没有满足指定候选约束")
        total=sum(weights[i][j] for i,j in enumerate(alternative))
        owner=next((owners[i] for i,j in enumerate(chosen) if j==column),None)
        result.append({"candidate_id":rival,"current_owner":owner,"weight_margin":round(baseline-total,5),
                       "assumption":"每篇舆论/每个雷达候选唯一归属；分数差不是校准概率"})
    return result


def close_rivals(edge,pool):
    return [e for e in pool if e["candidate_id"]!=edge["candidate_id"] and e["relation"]=="same"
            and set(e["platform_indices"]) & set(edge["platform_indices"])
            and abs(e["confidence"]-edge["confidence"])<.08]


def radar_batch_key(source):
    return source.get("sheet","radar"),source.get("batch") or source["id"]


def bundle_radar_evidence(groups,radars):
    """Preserve batch context without turning unassociated targets into event evidence."""
    byid={s["id"]:s for s in radars}
    batches={}
    for radar in radars:
        batches.setdefault(radar_batch_key(radar),[]).append(radar["id"])
    for group in groups:
        context=[]
        for rid in group["radar_ids"]:
            key=radar_batch_key(byid[rid])
            others=[sid for sid in batches[key] if sid not in group["radar_ids"]]
            if others:
                context.append({"selected":rid,"sheet":key[0],"batch":key[1],"other_candidates":others,
                    "reason":"仅核验已关联活动；同批号其他分支完整保留，未确认属于本事件，不自动构成反证"})
        group["radar_context"]=context
    return groups


def opinion_radar_corroboration(opinion,edge,selected):
    """Use facts absent from a vague news report to distinguish near-tied opinions."""
    details=[]
    for radar,i in selected:
        if i not in edge["platform_indices"] or not any(platform_similarity(p["name"],radar["platforms"][0]["name"])>=.9 for p in opinion["platforms"]):
            continue
        overlap=time_overlap(opinion,radar)
        missions=literal_missions(opinion);radar_missions=literal_missions(radar)
        time_bonus=6*overlap if overlap is not None and overlap>=.8 else 0
        mission_bonus=6 if len(missions)==len(radar_missions)==1 and set(missions)==set(radar_missions) else 0
        if time_bonus or mission_bonus:
            details.append({"opinion_id":opinion["id"],"radar_id":radar["id"],"platform_index":i,
                            "time_overlap":overlap,"mission_agreement":bool(mission_bonus),
                            "bonus":round(time_bonus+mission_bonus,5)})
    return max((d["bonus"] for d in details),default=0),details


def less_relevant_sibling(selected,rival,assigned):
    """A spatial hint may rank batch branches; it is never a new factual vote."""
    if selected["id"]==rival["id"] or radar_batch_key(selected)!=radar_batch_key(rival):
        return None
    if platform_similarity(selected["platforms"][0]["name"],rival["platforms"][0]["name"])<.9 or country(selected["platforms"][0].get("country"))!=country(rival["platforms"][0].get("country")):
        return None
    if set(literal_missions(selected))!=set(literal_missions(rival)) or (time_overlap(selected,rival) or 0)<.7:
        return None
    texts=[s for s in assigned if s["kind"] in {"news","opinion"} and s["location"].get("direction") in {"东海","南海","黄海"}]
    directions={s["location"]["direction"] for s in texts}
    if {s["kind"] for s in texts}!={"news","opinion"} or len(directions)!=1:
        return None
    selected_hint=set((selected.get("coordinate_hint") or {}).get("candidate_directions",[]))
    rival_hint=set((rival.get("coordinate_hint") or {}).get("candidate_directions",[]))
    if selected_hint!=directions or not rival_hint or rival_hint & directions:
        return None
    return {"candidate_id":rival["id"],"location_sources":[s["id"] for s in texts],
            "selected_hint":sorted(selected_hint),"other_hint":sorted(rival_hint),
            "reason":"两类文本的方向一致，仅使用未验证坐标排序同批候选；其他目标分支完整保留，不以批号相同制造本事件竞争或地点反证"}


def allocate(news_items, opinions, radars, alignments, recalls):
    op_edges = {n["id"]: {e["candidate_id"]:e for e in alignments[n["id"]]["opinion_edges"]} for n in news_items}
    opweights = []
    for n in news_items:
        heur = {r["id"]:r["score"] for r in recalls[n["id"]]["opinion"]}
        opweights.append([edge_weight(op_edges[n["id"]].get(o["id"]), heur.get(o["id"],0)) for o in opinions] + [0]*len(news_items))
    slots = [(n, i) for n in news_items for i in range(len(n["platforms"]))]
    rw = []
    for n,i in slots:
        edges = {e["candidate_id"]:e for e in alignments[n["id"]]["radar_edges"] if i in e["platform_indices"]}
        heur = {r["id"]:r["score"] for r in recalls[n["id"]]["radar"]}
        rw.append([edge_weight(edges.get(r["id"]), heur.get(r["id"],0)) for r in radars] + [0]*len(slots))
    ra = maximum_assignment(rw)
    selected={n["id"]:[] for n in news_items}
    for j,(n,i) in enumerate(slots):
        if ra[j]<len(radars) and rw[j][ra[j]]>0:
            selected[n["id"]].append((radars[ra[j]],i))
    joint_evidence={}
    for j,n in enumerate(news_items):
        if n.get("start_time") or n.get("end_time") or literal_missions(n):
            continue
        same=[e for e in op_edges[n["id"]].values() if e["relation"]=="same" and e["confidence"]>=.55]
        best=max((e["confidence"] for e in same),default=0)
        near={e["candidate_id"]:e for e in same if best-e["confidence"]<.08}
        if len(near)<2:
            continue
        for k,opinion in enumerate(opinions):
            if opinion["id"] not in near or opweights[j][k]<=0:
                continue
            bonus,details=opinion_radar_corroboration(opinion,near[opinion["id"]],selected[n["id"]])
            opweights[j][k]+=bonus
            if details:
                joint_evidence.setdefault(n["id"],[]).extend(details)
    opa = maximum_assignment(opweights)
    groups = [{"id": n["id"], "news_id": n["id"], "opinion_id": opinions[opa[j]]["id"] if opa[j] < len(opinions) and opweights[j][opa[j]]>0 else None,
               "radar_ids": [], "radar_slots": {}, "alignment": alignments[n["id"]], "candidates": recalls[n["id"]],
               "assignment_diagnostics":{"opinion":[],"radar":[]}}
              for j,n in enumerate(news_items)]
    byid = {g["id"]:g for g in groups}
    for nid,details in joint_evidence.items():
        byid[nid]["joint_assignment_evidence"]={"scope":"新闻缺少时段与任务时，以已选雷达与近分舆论的明示时段/任务协同分配；不消解任何明示反证","candidates":details}
    for j,group in enumerate(groups):
        if group["opinion_id"]:
            edge=op_edges[group["id"]][group["opinion_id"]]
            rivals=close_rivals(edge,list(op_edges[group["id"]].values()))
            group["assignment_diagnostics"]["opinion"]=assignment_competitors(opweights,opa,[s["id"] for s in opinions],
                [n["id"] for n in news_items],j,[e["candidate_id"] for e in rivals])
    for j,(n,i) in enumerate(slots):
        group = byid[n["id"]]
        group["radar_slots"][str(i)] = radars[ra[j]]["id"] if ra[j]<len(radars) and rw[j][ra[j]]>0 else None
        if group["radar_slots"][str(i)]:
            group["radar_ids"].append(group["radar_slots"][str(i)])
            edge=next(e for e in group["alignment"]["radar_edges"] if e["candidate_id"]==group["radar_slots"][str(i)])
            rivals=close_rivals(edge,group["alignment"]["radar_edges"])
            diagnostics=assignment_competitors(rw,ra,[s["id"] for s in radars],[s[0]["id"] for s in slots],j,
                                               [e["candidate_id"] for e in rivals if i in e["platform_indices"]])
            group["assignment_diagnostics"]["radar"].extend(dict(d,platform_index=i) for d in diagnostics)
    return bundle_radar_evidence(groups,radars)


def linked_platforms(group, news, opinion, radars, platform_index):
    result = [(news["id"], news["platforms"][platform_index])]
    if opinion:
        edge = next((e for e in group["alignment"]["opinion_edges"] if e["candidate_id"]==opinion["id"]), None)
        if edge and platform_index in edge["platform_indices"]:
            matches = [(platform_similarity(news["platforms"][platform_index]["name"],p["name"]),p) for p in opinion["platforms"]]
            # Only attach quantities to an identified platform, never copy a formation count.
            score, p = max(matches, key=lambda x:x[0])
            if score >= .45:
                result.append((opinion["id"],p))
    rid = group["radar_slots"].get(str(platform_index))
    if rid:
        radar = next(r for r in radars if r["id"]==rid)
        # Multiple explicitly assigned observations may be checked, but batch equality is not assignment.
        indices={e["candidate_id"] for e in group["alignment"]["radar_edges"] if platform_index in e["platform_indices"]}
        result.extend((r["id"],r["platforms"][0]) for r in radars if r["id"]==rid or r["id"] in indices)
    return result


def counterevidence_hint(news, opinion, selected_radars, news_items, opinions):
    """A unique literal identity and radar clock can justify a provisional counter-pair, never trust."""
    relation=location_relation(news["location"],opinion["location"])
    if len(news["platforms"])!=1 or relation not in {"different_direction","different_named_regions"}:
        return None
    platform=news["platforms"][0]
    def mission_key(s):
        return compact(s["mission"].get("value") or "").removesuffix("任务")
    mission=mission_key(news)
    if not platform.get("country") or not mission or mission in {compact(v) for v in ["活动","持续活动","行动","机动","保持机动"]}:
        return None
    def identical(s):
        if len(s["platforms"])!=1:
            return False
        p=s["platforms"][0]
        return (platform_similarity(platform["name"],p["name"])>=.9 and country(p.get("country"))==country(platform["country"])
                and mission_key(s)==mission)
    if not identical(opinion) or sum(identical(s) for s in news_items)!=1 or sum(identical(s) for s in opinions)!=1:
        return None
    if platform.get("quantity") is not None and opinion["platforms"][0].get("quantity") is not None and platform["quantity"]!=opinion["platforms"][0]["quantity"]:
        return None
    if not news.get("start_time") or not news.get("end_time"):
        return None
    overlap=time_overlap(news,opinion)
    if overlap is not None and overlap<.8:
        return None
    clocks=[r["id"] for r in selected_radars if identical(r) and (time_overlap(news,r) or 0)>=.8]
    if not clocks:
        return None
    return {"same_day_scope":True,"unique_platform_country_mission_in_both_text_pools":True,
            "radar_clock_matches":clocks,"location_relation":relation,"certainty":"provisional_only"}


def review_unmatched_opinions(client, groups, news_items, opinions, radars, alignments, recalls):
    """Reassess rejected counter-evidence without forcing a leftover pairing."""
    used={g["opinion_id"] for g in groups if g["opinion_id"]}
    remaining=[s for s in opinions if s["id"] not in used]
    radar_byid={s["id"]:s for s in radars}
    news_byid={s["id"]:s for s in news_items}
    instructions="""现在进行全局分配后的剩余材料复核。重点检查前次是否把相关反证误判为不同事件。
地点/国籍/数量/任务某一字段有冲突，本身不能充分证明different；冲突可能是赛题中的受污染字段。
综合比较型号、国籍、数量、日期、任务、其他同型号新闻、新闻与已关联雷达的活动时间。
同型号另有活动时要保留歧义；仅因是剩余材料绝不能强行配对。没有充分关联则different或uncertain。
若其他身份和活动线索充分吻合、只剩明确地点冲突，可same，reason必须保留地点反证，可信性留待核验。
只有日期的舆论不能当作与具体时刻不一致。对每篇候选舆论必须明确给出same/uncertain/different，不得省略。
已关联雷达只作为上下文，无需重新输出雷达关联。输出关联，不要纠正任何原文错误。"""
    for group in groups:
        if group["opinion_id"] or not remaining:
            continue
        news=news_byid[group["news_id"]]
        compatible=[s for s in remaining if any(platform_similarity(p["name"],q["name"])>=.45
                      for p in news["platforms"] for q in s["platforms"])]
        scores={r["id"]:r["score"] for r in recalls[news["id"]]["opinion"]}
        compatible=sorted(compatible,key=lambda s:(-scores.get(s["id"],0),s["id"]))[:3]
        if not compatible:
            continue
        other_news=[public_view(s) for s in news_items if s["id"]!=news["id"] and
                    any(platform_similarity(p["name"],q["name"])>=.45 for p in news["platforms"] for q in s["platforms"])]
        review=align_source(client,news,compatible,[],
                            {"review":"全局分配后未匹配材料的反证复核","other_same_platform_news":other_news,
                             "already_associated_radars":[public_view(radar_byid[rid]) for rid in group["radar_ids"]],
                             "previous_opinion_edges":alignments[news["id"]]["opinion_edges"]},instructions)
        edges={e["candidate_id"]:e for e in alignments[news["id"]]["opinion_edges"]}
        byid={s["id"]:s for s in compatible}
        for edge in review["opinion_edges"]:
            if edge["relation"]=="different":
                hint=counterevidence_hint(news,byid[edge["candidate_id"]],[radar_byid[rid] for rid in group["radar_ids"]],news_items,opinions)
                if hint:
                    edge["model_proposal"]=dict(edge)
                    edge.update({"relation":"uncertain","confidence":.65,"platform_indices":[0],"association_check":hint,
                                 "reason":edge["reason"]+"；程序保留待复核关联：同日双方唯一的型号/国籍/明确任务组合且新闻与雷达时段吻合；地点反证保留，不能据此判可信"})
            edge["reviewed_after_unmatched"]=True
            edges[edge["candidate_id"]]=edge
        alignments[news["id"]]["opinion_edges"]=list(edges.values())
        alignments[news["id"]]["unmatched_review"]=review
        recalls[news["id"]]["unmatched_review"]=True
        print(f"[counter-review] {news['id']} remaining opinions={len(compatible)}",flush=True)
    return allocate(news_items,opinions,radars,alignments,recalls)


def reconcile_model_issues(verification):
    """Keep model proposals, but recheck factual issues against the audited source matrix."""
    import re
    aliases={"quantity":["quantity","数量","架数","架次","艘数"],
             "country":["country","国籍","国家"],"platform":["platform","平台身份","型号"],
             "location":["location","地点","海域","区域","位置"],
             "time":["time","时间","时刻","起点","终点"],
             "mission":["mission","任务","科目"],"movement":["movement","进出","进入","离开"]}
    proposed=list(verification.get("unresolved_issues",[]))
    remaining=[];reconciled=[]
    for issue in proposed:
        fields={field for field,words in aliases.items() if any(word in issue for word in words)}
        facts=[f for f in verification["facts"] if f["field"] in fields]
        if any(f["status"]=="conflict" for f in facts):
            reconciled.append({"model_issue":issue,"fields":sorted(fields),
                               "decision":"字段反证由程序事实矩阵保留；不采用模型多数票消解或重复自由文本说明"})
        elif re.search(r"缺失|未提|未说明|未明确|仅由|只有|不足|单一来源|仅一类",issue) and facts and all(
                f["status"]=="supported" and len(f.get("supporting_source_types",[]))>=2 for f in facts):
            reconciled.append({"model_issue":issue,"fields":sorted(fields),
                               "decision":"事实已获至少两类来源印证；某一来源缺字段不再作为事件阻断原因"})
        elif fields=={"quantity"} and facts and all(f["status"]=="insufficient" for f in facts) and re.search(r"缺失|未提|仅|不足|未明确",issue) and not re.search(r"关联|归属|同一事件|另一活动|歧义",issue):
            reconciled.append({"model_issue":issue,"fields":["quantity"],
                               "decision":"数量继续列为待核实且不纳入已验证事实；仅数量未印证不阻断已关联的活动"})
        else:
            remaining.append(issue)
    verification["model_unresolved_issues"]=proposed
    verification["unresolved_issues"]=remaining
    verification["reconciled_model_issues"]=reconciled


def apply_guards(group, news, opinion, radars, verification, radar_candidates=None):
    assigned = [news] + ([opinion] if opinion else []) + radars
    radar_byid={s["id"]:s for s in (radar_candidates or radars)}
    kinds = {s["id"]:s["kind"] for s in assigned}
    blockers = []
    if not opinion:
        blockers.append("缺少已确认关联的舆论来源")
    if not radars:
        blockers.append("缺少已确认关联的雷达来源")
    if any(not rid for rid in group["radar_slots"].values()):
        blockers.append("联合活动有平台缺少已关联雷达活动")
    if not verification["same_event"]:
        blockers.append("模型未确认三源同一事件关联")
    for conflict in group.get("batch_conflicts",[]):
        blockers.append(f"雷达批号{conflict['batch']}跨组归属未消解：{conflict['owners']}")
    # Matrix rechecks do not depend on the model noticing contradictory task/clock/movement values.
    check_common_facts(verification,assigned)
    check_location_fact(verification,assigned)
    # Recheck explicit platform, nationality and counts from immutable source extraction.
    # Preserve the model proposal; source omissions never become votes or counter-evidence.
    for i,p in enumerate(news["platforms"]):
        linked=linked_platforms(group,news,opinion,radars,i)
        for name in ["platform","country","quantity"]:
            fact=next(f for f in verification["facts"] if f["platform_index"]==i and f["field"]==name)
            proposal=dict(fact)
            if name=="platform":
                support=[sid for sid,value in linked if platform_similarity(value["name"],p["name"])>=.45]
                conflicts=[]
                claim=p["name"]
                values={sid:value["name"] for sid,value in linked}
                for sid,value in linked:
                    if kinds[sid]=="radar" and platform_similarity(value["name"],p["name"])<.45:
                        blockers.append(f"平台{i}雷达身份与新闻身份无法完成字面核对：{value['name']} / {p['name']}")
            else:
                values={sid:country(value.get(name)) if name=="country" else value.get(name) for sid,value in linked if value.get(name) is not None}
                if values:
                    totals={value:len({kinds[sid] for sid,v in values.items() if v==value}) for value in values.values()}
                    chosen=max(totals,key=lambda value:(totals[value],str(value)))
                    support=[sid for sid,value in values.items() if value==chosen]
                    conflicts=[sid for sid,value in values.items() if value!=chosen]
                    claim=str(chosen)+("架" if "飞机" in [r["sheet"] for r in radars] else "艘") if name=="quantity" else str(chosen)
                else:
                    support,conflicts,claim=[],[],"未说明"
            status="conflict" if conflicts else "supported" if len({kinds[sid] for sid in support})>=2 else "insufficient"
            fact.update({"claim":claim,"supporting_sources":support,"conflicting_sources":conflicts,"status":status,
                         "reason":f"程序按已关联平台的明示字段复核：{values}；缺失不支持也不反对",
                         "model_proposal":proposal,"checked_by":"explicit_source_matrix"})
    for field in verification["facts"]:
        field["supporting_sources"] = sorted(set(field["supporting_sources"]))
        field["conflicting_sources"] = sorted(set(field["conflicting_sources"]))
        rejected = []
        for sid in field["supporting_sources"]:
            source = next(s for s in assigned if s["id"]==sid)
            name = field["field"]
            if name=="location":
                # An unverified coordinate box is not an independent region assertion.
                target = normalize_location(field["claim"])
                loc = source["location"]
                relation = location_relation(target,loc)
                valid = relation=="same" or (target["precision"]=="大范围" and relation in {"contains","same_direction_only"})
                if loc.get("region") and compact(loc["region"]) in compact(field["claim"]):
                    valid = True
                if not valid:
                    rejected.append(sid)
            elif name in {"country","quantity","platform"} and field["platform_index"]>=0:
                linked = dict(linked_platforms(group,news,opinion,radars,field["platform_index"]))
                if sid not in linked or (name!="platform" and linked[sid].get(name) is None):
                    rejected.append(sid)
                elif name=="platform" and platform_similarity(linked[sid]["name"],field["claim"]) < .45:
                    rejected.append(sid)
                elif name=="quantity":
                    import re
                    values=quantity_numbers(field["claim"])
                    if not values and re.fullmatch(r"\d+",field["claim"].strip()):
                        values={int(field["claim"])}
                    if linked[sid]["quantity"] not in values:
                        rejected.append(sid)
                elif name=="country" and compact(country(linked[sid]["country"])) not in compact(field["claim"]):
                    rejected.append(sid)
            elif name=="mission" and source["mission"].get("value") in {None,"活动","持续活动","任务行动","行动","机动","保持机动"}:
                rejected.append(sid)
            elif name=="movement" and source.get("movement") in {None,"未说明"}:
                rejected.append(sid)
            elif name=="time" and field.get("checked_by")!="literal_assertion_matrix":
                import re
                precise = bool(re.search(r"\d{1,2}[:：时]\d{1,2}",field["claim"]))
                if precise and not (source.get("start_time") and source.get("end_time")):
                    rejected.append(sid)
                elif precise:
                    clocks={int(h)*60+int(m) for h,m in re.findall(r"(\d{1,2})[:：时](\d{1,2})",field["claim"])}
                    observed={minutes(source["start_time"]),minutes(source["end_time"])}
                    if not clocks <= observed:
                        rejected.append(sid)
        if rejected:
            field["rejected_supports"] = rejected
            field["supporting_sources"] = [sid for sid in field["supporting_sources"] if sid not in rejected]
            field["reason"] += f"；程序剔除缺失或精度不足的支持引用：{rejected}"
        support_kinds = {kinds[sid] for sid in field["supporting_sources"]}
        field["supporting_source_types"] = sorted(support_kinds)
        if field["status"]=="supported" and len(support_kinds)<2:
            field["status"]="insufficient"
            field["reason"] += "；程序检查：不足两类独立来源支持"
        if field["conflicting_sources"] or field["status"]=="conflict":
            blockers.append(f"{field['field']}关键事实冲突：{field['reason']}")
    for radar in radars:
        if radar["anomalies"]:
            blockers.extend(f"{radar['id']}：{a}" for a in radar["anomalies"])
    edges = group["alignment"]
    selected_edges = [e for e in edges["opinion_edges"] if e["candidate_id"]==group["opinion_id"]] + [e for e in edges["radar_edges"] if e["candidate_id"] in group["radar_ids"]]
    for edge in selected_edges:
        if edge["relation"] != "same":
            blockers.append(f"{edge['candidate_id']}活动归属仍有歧义：{edge['reason']}")
    # Close competing associations are preserved as uncertainty, not silently discarded.
    for edge in selected_edges:
        pool = edges["opinion_edges"] if edge["candidate_id"].startswith("O") else edges["radar_edges"]
        rivals=close_rivals(edge,pool)
        kind="opinion" if edge["candidate_id"].startswith("O") else "radar"
        diagnostics=group.get("assignment_diagnostics",{}).get(kind,[])
        unresolved=[]
        for rival in rivals:
            if kind=="radar" and rival["candidate_id"] in group["radar_ids"]:
                # A bundled branch is retained evidence of the same batch, not an alternative event.
                continue
            if kind=="radar":
                preferred=less_relevant_sibling(next(r for r in radars if r["id"]==edge["candidate_id"]),
                    radar_byid[rival["candidate_id"]],assigned) if rival["candidate_id"] in radar_byid else None
                if preferred:
                    group.setdefault("resolved_branch_selection",[]).append({"selected":edge["candidate_id"],**preferred})
                    continue
            proof=next((d for d in diagnostics if d["candidate_id"]==rival["candidate_id"] and d["current_owner"]
                        and d["current_owner"]!=group.get("id") and d["weight_margin"]>=8),None)
            if proof:
                group.setdefault("resolved_competition",[]).append({"selected":edge["candidate_id"],**proof})
            else:
                unresolved.append(rival["candidate_id"])
        if unresolved:
            blockers.append(f"{edge['candidate_id']}存在未被全局分配消解的同活动候选：{unresolved}")
    supported = {(f["platform_index"],f["field"]) for f in verification["facts"] if f["status"]=="supported"}
    for i,p in enumerate(news["platforms"]):
        if (i,"platform") not in supported:
            blockers.append(f"平台{i} {p['name']}身份未获得两类来源印证")
        country_fact=next(f for f in verification["facts"] if f["platform_index"]==i and f["field"]=="country")
        if p.get("country") and (i,"country") not in supported and country_fact["status"]!="conflict":
            blockers.append(f"平台{i} {p['name']}国籍未获得两类来源印证")
    if (-1,"location") not in supported and not any(f["field"]=="location" and f["status"]=="conflict" for f in verification["facts"]):
        blockers.append("活动地点未获得两类来源在明确精度上的印证")
    reconcile_model_issues(verification)
    blockers += verification["unresolved_issues"]
    group["verification"] = verification
    group["reasons"] = list(dict.fromkeys(blockers))
    group["status"] = "可信事件" if not blockers else "不可信事件"
    verification["model_explanation"]=verification["explanation"]
    verification["explanation"]=("三类来源已关联，纳入结论的事实获得至少两类独立来源支持，未发现未消解明示反证。"
                                 if not blockers else "程序核验未通过："+"；".join(group["reasons"]))
    group["verified_facts"] = [f for f in verification["facts"] if f["status"]=="supported"]
    group["radar_row_count"] = sum(r["row_count"] for r in radars)
    group["direction"] = news["location"].get("direction")
    group["report_eligible"] = group["status"]=="可信事件" and group["direction"] in {"东海","南海","黄海"}
    return group


def match(client, texts, radars, output):
    news_items = [s for s in texts if s["kind"]=="news"]
    opinions = [s for s in texts if s["kind"]=="opinion"]
    sources = texts + radars
    byid = {s["id"]:s for s in sources}
    positions = {s["id"]:i for i,s in enumerate(sources)}
    similarities, vector_manifest = build_index(client, sources, output)
    print(f"[index] {len(sources)} BGE-M3 vectors generated and stored", flush=True)
    alignments, recalls = {}, {}
    for j,news in enumerate(news_items,1):
        op, oscores = recall(news,opinions,similarities,positions)
        ra, rscores = recall(news,radars,similarities,positions)
        recalls[news["id"]] = {"opinion":oscores,"radar":rscores}
        alignments[news["id"]] = align_source(client,news,op,ra,recalls[news["id"]])
        dump(output / "运行记录" / "关联进度.json", {"alignments":alignments,"recalls":recalls})
        print(f"[align] {j}/{len(news_items)} {news['id']} candidates opinion={len(op)}, radar={len(ra)}", flush=True)
    groups = allocate(news_items,opinions,radars,alignments,recalls)
    # Missing-source/uncertain groups receive one expanded recall, not an unbounded agent loop.
    review_ids = [g["id"] for g in groups if not g["opinion_id"] or any(r is None for r in g["radar_slots"].values())]
    for nid in review_ids:
        news = byid[nid]
        op, oscores = recall(news,opinions,similarities,positions,full_scores=True)
        ra, rscores = recall(news,radars,similarities,positions,full_scores=True)
        # The whole day has already been indexed; avoid overflowing the model context with all 50 texts.
        g = next(g for g in groups if g["id"]==nid)
        if not g["opinion_id"]:
            op = expanded_candidates(news,opinions,oscores)
        if any(r is None for r in g["radar_slots"].values()):
            ra = expanded_candidates(news,radars,rscores)
        review = align_source(client,news,op,ra,{"review": "缺源扩大检索，保留反证，不强行凑组", **recalls[nid]})
        alignments[nid] = review
        recalls[nid]["expanded_review"] = True
        print(f"[review] {nid} expanded recall", flush=True)
    if review_ids:
        groups = allocate(news_items,opinions,radars,alignments,recalls)
    groups=review_unmatched_opinions(client,groups,news_items,opinions,radars,alignments,recalls)
    dump(output / "运行记录" / "全局分配.json", groups)
    for j,group in enumerate(groups,1):
        news = byid[group["news_id"]]
        opinion = byid.get(group["opinion_id"])
        selected_radars = [byid[rid] for rid in group["radar_ids"]]
        sibling_ids = {sid for radar in selected_radars for sid in radar["sibling_ids"]} - set(group["radar_ids"])
        verification = verify_group(client,news,opinion,selected_radars,group["alignment"],[byid[s] for s in sorted(sibling_ids)])
        apply_guards(group,news,opinion,selected_radars,verification,radar_candidates=radars)
        dump(output / "运行记录" / "核验进度.json", groups[:j])
        print(f"[verify] {j}/{len(groups)} {group['id']} {group['status']} radar_rows={group['radar_row_count']}", flush=True)
    return groups, vector_manifest
