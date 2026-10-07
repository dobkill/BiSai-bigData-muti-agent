import json
import sqlite3

import numpy as np

from .agents import align_source, quantity_numbers, verify_group
from .assignment import maximum_assignment
from .data import compact, country, minutes, platform_similarity, public_view, time_overlap
from .geography import location_relation, normalize_location
from .ollama_client import dump
from .literal_facts import check_common_facts, check_location_fact


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


def recall(news, pool, similarities, positions):
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
    return [byid[s["id"]] for s in records], records


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


def allocate(news_items, opinions, radars, alignments, recalls):
    op_edges = {n["id"]: {e["candidate_id"]:e for e in alignments[n["id"]]["opinion_edges"]} for n in news_items}
    opweights = []
    for n in news_items:
        heur = {r["id"]:r["score"] for r in recalls[n["id"]]["opinion"]}
        opweights.append([edge_weight(op_edges[n["id"]].get(o["id"]), heur.get(o["id"],0)) for o in opinions] + [0]*len(news_items))
    opa = maximum_assignment(opweights)
    slots = [(n, i) for n in news_items for i in range(len(n["platforms"]))]
    rw = []
    for n,i in slots:
        edges = {e["candidate_id"]:e for e in alignments[n["id"]]["radar_edges"] if i in e["platform_indices"]}
        heur = {r["id"]:r["score"] for r in recalls[n["id"]]["radar"]}
        rw.append([edge_weight(edges.get(r["id"]), heur.get(r["id"],0)) for r in radars] + [0]*len(slots))
    ra = maximum_assignment(rw)
    groups = [{"id": n["id"], "news_id": n["id"], "opinion_id": opinions[opa[j]]["id"] if opa[j] < len(opinions) and opweights[j][opa[j]]>0 else None,
               "radar_ids": [], "radar_slots": {}, "alignment": alignments[n["id"]], "candidates": recalls[n["id"]],
               "assignment_diagnostics":{"opinion":[],"radar":[]}}
              for j,n in enumerate(news_items)]
    byid = {g["id"]:g for g in groups}
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
    return groups


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
        result.append((rid,radar["platforms"][0]))
    return result


def counterevidence_hint(news, opinion, selected_radars, news_items, opinions):
    """A unique literal identity and radar clock can justify a provisional counter-pair, never trust."""
    if len(news["platforms"])!=1 or location_relation(news["location"],opinion["location"])!="different_direction":
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
            "radar_clock_matches":clocks,"location_relation":"different_direction","certainty":"provisional_only"}


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
                    edge.update({"relation":"uncertain","confidence":.65,"association_check":hint,
                                 "reason":edge["reason"]+"；程序保留待复核关联：同日双方唯一的型号/国籍/明确任务组合且新闻与雷达时段吻合；地点反证保留，不能据此判可信"})
            edge["reviewed_after_unmatched"]=True
            edges[edge["candidate_id"]]=edge
        alignments[news["id"]]["opinion_edges"]=list(edges.values())
        alignments[news["id"]]["unmatched_review"]=review
        recalls[news["id"]]["unmatched_review"]=True
        print(f"[counter-review] {news['id']} remaining opinions={len(compatible)}",flush=True)
    return allocate(news_items,opinions,radars,alignments,recalls)


def apply_guards(group, news, opinion, radars, verification):
    assigned = [news] + ([opinion] if opinion else []) + radars
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
    if (-1,"location") not in supported:
        blockers.append("活动地点未获得两类来源在明确精度上的印证")
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
        op, oscores = recall(news,opinions,similarities,positions)
        ra, rscores = recall(news,radars,similarities,positions)
        # Expand to the full same-day pool for missing source. Keep other source candidates bounded.
        g = next(g for g in groups if g["id"]==nid)
        if not g["opinion_id"]:
            op = opinions
        if any(r is None for r in g["radar_slots"].values()):
            ra = radars
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
        apply_guards(group,news,opinion,selected_radars,verification)
        dump(output / "运行记录" / "核验进度.json", groups[:j])
        print(f"[verify] {j}/{len(groups)} {group['id']} {group['status']} radar_rows={group['radar_row_count']}", flush=True)
    return groups, vector_manifest
