import itertools
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from event_matcher.agents import exact_quote, quantity_numbers
from event_matcher.assignment import maximum_assignment
from event_matcher.data import country, load_radar, load_texts, platform_similarity, platform_tokens
from event_matcher.export import export_results
from event_matcher.geography import location_relation, normalize_location
from event_matcher.literal_facts import check_common_facts, repair_source_assertions
from event_matcher.matching import allocate, apply_guards, counterevidence_hint, review_unmatched_opinions


class MatchingTests(unittest.TestCase):
    def test_global_assignment_against_exhaustive_search(self):
        rng=random.Random(31)
        for n in range(1,5):
            for _ in range(15):
                weights=[[rng.randrange(-5,8) for _ in range(n+1)] for _ in range(n)]
                chosen=maximum_assignment(weights)
                actual=sum(weights[i][j] for i,j in enumerate(chosen))
                best=max(sum(weights[i][j] for i,j in enumerate(p)) for p in itertools.permutations(range(n+1),n))
                self.assertEqual(actual,best)
                self.assertEqual(len(set(chosen)),n)

    def test_abstention_better_than_incorrect_forced_pair(self):
        self.assertEqual(maximum_assignment([[-10,0]]),[1])

    def test_unmatched_review_preserves_counterevidence_without_forcing_pair(self):
        import copy
        def source(sid,kind,place):
            return {"id":sid,"kind":kind,"platforms":[{"name":"E-2C预警机","country":"法国","quantity":2}],
                    "location":normalize_location(place),"mission":{"value":"空中警戒"}}
        news=source("N001","news","仁爱礁附近海域")
        opinion=source("O001","opinion","赤尾屿海域")
        radar=source("R001","radar",None)
        original={"opinion_edges":[{"candidate_id":"O001","platform_indices":[0],"relation":"different","confidence":.95,"reason":"地点不同"}],
                  "radar_edges":[{"candidate_id":"R001","platform_indices":[0],"relation":"same","confidence":.95,"reason":"身份时间相符"}],"ambiguity":""}
        recalls={"N001":{"opinion":[{"id":"O001","score":4}],"radar":[{"id":"R001","score":4}]}}
        for verdict in ["different","same"]:
            with self.subTest(verdict=verdict):
                class ReviewClient:
                    def chat(self,role,prompt,payload,schema,validate,**kwargs):
                        result=schema.model_validate({"opinion_edges":[{"candidate_id":"O001","platform_indices":[0],
                            "relation":verdict,"confidence":.85,"reason":"地点明确冲突，其余活动线索匹配"}],"radar_edges":[],"ambiguity":""})
                        validate(result)
                        return result
                alignments={"N001":copy.deepcopy(original)}
                groups=allocate([news],[opinion],[radar],alignments,recalls)
                self.assertIsNone(groups[0]["opinion_id"])
                groups=review_unmatched_opinions(ReviewClient(),groups,[news],[opinion],[radar],alignments,copy.deepcopy(recalls))
                self.assertEqual(groups[0]["opinion_id"],"O001" if verdict=="same" else None)
                self.assertEqual(groups[0]["radar_ids"],["R001"])
                edge=groups[0]["alignment"]["opinion_edges"][0]
                self.assertIn("地点明确冲突",edge["reason"])
                self.assertTrue(edge["reviewed_after_unmatched"])

    def test_provisional_counter_pair_requires_unique_identity_and_radar_clock(self):
        import copy
        def source(sid,kind,place,mission,start,end):
            return {"id":sid,"kind":kind,"platforms":[{"name":"E-2C预警机","country":"法国","quantity":2}],
                    "location":normalize_location(place),"mission":{"value":mission},"start_time":start,"end_time":end}
        news=source("N001","news","仁爱礁附近海域","空中警戒","05:01","07:26")
        opinion=source("O001","opinion","赤尾屿海域","空中警戒任务",None,None)
        radar=source("R001","radar",None,"空中警戒","05:01","07:26")
        self.assertIsNotNone(counterevidence_hint(news,opinion,[radar],[news],[opinion]))
        second=copy.deepcopy(opinion);second["id"]="O002"
        self.assertIsNone(counterevidence_hint(news,opinion,[radar],[news],[opinion,second]))
        second_news=copy.deepcopy(news);second_news["id"]="N002"
        self.assertIsNone(counterevidence_hint(news,opinion,[radar],[news,second_news],[opinion]))
        radar["start_time"],radar["end_time"]="15:01","17:26"
        self.assertIsNone(counterevidence_hint(news,opinion,[radar],[news],[opinion]))

    def test_geography_precision_and_transit(self):
        detailed=normalize_location("永暑礁附近")
        broad=normalize_location("南海方向")
        other=normalize_location("黄岩岛海域")
        self.assertEqual(detailed["region"],"永暑礁附近海域")
        self.assertEqual(location_relation(detailed,broad),"contains")
        self.assertEqual(location_relation(detailed,other),"same_direction_only")
        self.assertEqual(normalize_location("宫古水道")["direction"],"其他")
        self.assertIsNone(normalize_location(None)["direction"])
        self.assertEqual(location_relation(normalize_location("中太平洋海域"),normalize_location("中太平洋")),"same")
        self.assertEqual(normalize_location("夏威夷附近海空域")["region"],"夏威夷附近")
        self.assertEqual(country("意大利空军"),"意大利")
        self.assertEqual(normalize_location("阿拉伯海围绕任务海域")["region"],"阿拉伯海")
        self.assertEqual(normalize_location("安达曼海任务海域")["region"],"安达曼海")

    def test_platform_role_does_not_become_mission(self):
        source={"id":"N001","kind":"news","text":"标题：中国运-8反潜巡逻机持续活动\n正文：中国两架运-8反潜巡逻机在南海北部持续活动，围绕任务区域保持机动。",
                "mission":{"value":"反潜巡逻","quote":"运-8反潜巡逻机"},"movement":"活动","extraction_repairs":[]}
        repair_source_assertions(source)
        self.assertIsNone(source["mission"]["value"])
        self.assertEqual(source["extraction_model_proposal"]["mission"]["value"],"反潜巡逻")

    def test_movement_requires_actual_entry_verb(self):
        source={"id":"O001","kind":"opinion","text":"标题：韩国直升机执行反潜巡逻\n正文：韩国AW-159在阿拉伯海组织反潜巡逻，按计划保持机动。",
                "mission":{"value":"反潜巡逻","quote":"组织反潜巡逻"},"movement":"进入","movement_quote":"保持机动","extraction_repairs":[]}
        repair_source_assertions(source)
        self.assertEqual(source["movement"],"活动")
        self.assertEqual(source["extraction_model_proposal"]["movement"]["value"],"进入")

    def test_different_explicit_tasks_are_not_merged_by_model(self):
        def source(sid,kind,task):
            return {"id":sid,"kind":kind,"text":"正文：平台组织"+task+"。","mission":{"value":task},"start_time":None,"end_time":None}
        assigned=[source("N001","news","对海突击训练"),source("O001","opinion","战备警巡"),source("R001","radar","对海突击训练")]
        verify={"facts":[{"platform_index":-1,"field":"mission","claim":"对海突击训练/战备警巡","status":"supported"}]}
        check_common_facts(verify,assigned)
        fact=next(f for f in verify["facts"] if f["field"]=="mission")
        self.assertEqual(fact["status"],"conflict")
        self.assertEqual(fact["conflicting_sources"],["O001"])
        self.assertNotIn("/",fact["claim"])

    def test_single_endpoint_countervalue_is_not_ignored(self):
        assigned=[]
        for sid,kind,start,end in [("N001","news","21:31","22:12"),("O001","opinion","20:47",None),("R001","radar","20:47","22:12")]:
            assigned.append({"id":sid,"kind":kind,"text":"正文：平台组织预警指挥。","mission":{"value":"预警指挥"},"start_time":start,"end_time":end})
        verify={"facts":[]}
        check_common_facts(verify,assigned)
        fact=next(f for f in verify["facts"] if f["field"]=="time")
        self.assertEqual(fact["status"],"conflict")
        self.assertEqual(fact["components"]["start_time"]["conflicts"],["N001"])
        self.assertEqual(fact["components"]["end_time"]["status"],"supported")

    def test_activity_does_not_corroborate_specific_entry(self):
        assigned=[{"id":"N001","kind":"news","text":"正文：平台进入东海。","mission":{"value":None}},
                  {"id":"O001","kind":"opinion","text":"正文：平台在东海持续活动。","mission":{"value":None}},
                  {"id":"R001","kind":"radar","mission":{"value":None}}]
        verify={"facts":[]}
        check_common_facts(verify,assigned)
        fact=next(f for f in verify["facts"] if f["field"]=="movement")
        self.assertEqual(fact["claim"],"活动")
        self.assertEqual(fact["source_values"]["N001"],"进入")

    def test_platform_spacing_does_not_hide_type(self):
        self.assertEqual(platform_tokens("歼 -35 舰载隐身战斗机"),platform_tokens("歼-35舰载隐身战斗机"))
        self.assertEqual(platform_similarity("F-35B","F-15SG"),0)

    def test_explicit_quantity_and_original_quote(self):
        raw="韩国AW-159反潜直升机从11时37分至13时02分活动。"
        self.assertEqual(exact_quote(raw,"11 时 37 分至 13 时 02 分"),"11时37分至13时02分")
        self.assertEqual(quantity_numbers("韩国AW-159反潜直升机"),set())
        self.assertEqual(quantity_numbers("韩国两架AW-159反潜直升机"),{2})

    def test_radar_partition_preserves_every_original_row(self):
        day=Path(__file__).resolve().parents[1]/"Datas"/"数据"/"2025.5.1"
        radar,manifest=load_radar(day)
        keys=[(a["sheet"],r["row"]) for a in radar for r in a["records"]]
        self.assertEqual(len(keys),600)
        self.assertEqual(len(set(keys)),600)
        self.assertEqual(manifest["batches"],52)
        self.assertEqual(len(radar),54)
        split=[a for a in radar if a["sibling_ids"]]
        self.assertEqual(len(split),4)
        self.assertTrue(all(a["anomalies"] for a in split))

    def test_majority_cannot_override_quantity_conflict(self):
        def source(sid,kind,qty):
            return {"id":sid,"kind":kind,"platforms":[{"name":"903A型综合补给舰","country":"中国","quantity":qty}],
                    "location":normalize_location("万安滩海域"),"mission":{"value":"远海训练"},"start_time":None,"end_time":None,"movement":"活动"}
        news,opinion,radar=source("N001","news",1),source("O001","opinion",1),source("R001","radar",2)
        radar.update({"sheet":"舰船","anomalies":[],"row_count":10})
        edge=lambda sid:{"candidate_id":sid,"platform_indices":[0],"relation":"same","confidence":.95,"reason":"同活动"}
        group={"opinion_id":"O001","radar_ids":["R001"],"radar_slots":{"0":"R001"},"alignment":{"opinion_edges":[edge("O001")],"radar_edges":[edge("R001")]}}
        facts=[{"platform_index":i,"field":field,"claim":claim,"supporting_sources":["N001","O001"],"conflicting_sources":[],"status":"supported","reason":"两源一致"}
               for i,field,claim in [(0,"platform","903A"),(0,"country","中国"),(0,"quantity","1艘"),(-1,"location","万安滩海域")]]
        verify={"same_event":True,"facts":facts,"unresolved_issues":[],"explanation":"模型误按多数票"}
        result=apply_guards(group,news,opinion,[radar],verify)
        self.assertEqual(result["status"],"不可信事件")
        self.assertTrue(any("quantity" in reason for reason in result["reasons"]))

    def test_export_roundtrip_preserves_all_text_bytes_and_radar_fields(self):
        day=Path(__file__).resolve().parents[1]/"Datas"/"数据"/"2025.5.1"
        texts=load_texts(day)
        radar,rmanifest=load_radar(day)
        for s in texts:
            s["location"]=normalize_location(None)
        groups=[]
        for s in texts:
            if s["kind"]!="news":
                continue
            groups.append({"id":s["id"],"news_id":s["id"],"opinion_id":None,"radar_ids":[],"radar_slots":{},
                           "status":"不可信事件","radar_row_count":0,"direction":None,"report_eligible":False,
                           "reasons":["仅用于验证导出，不进行模型推理"],"verification":{"explanation":"测试缺源组","facts":[]},
                           "alignment":{"opinion_edges":[],"radar_edges":[]},"candidates":{"opinion":[],"radar":[]}})
        with tempfile.TemporaryDirectory(prefix="matching-export-") as tmp:
            result=export_results(groups,texts,radar,rmanifest,Path(tmp),SimpleNamespace(input=day,model="test",embedding_model="test"),{},SimpleNamespace(calls=0,hits=0))
            self.assertTrue(result["integrity"]["passed"])
            self.assertEqual(result["integrity"]["exported_texts"],100)
            self.assertEqual(result["integrity"]["exported_radar_rows"],600)
            opinion=next(s for s in texts if s["kind"]=="opinion")
            formerly_unmatched=Path(tmp)/"未匹配材料"/"文本"/(opinion["id"]+"_"+opinion["filename"])
            self.assertTrue(formerly_unmatched.exists())
            groups[0]["opinion_id"]=opinion["id"]
            result=export_results(groups,texts,radar,rmanifest,Path(tmp),SimpleNamespace(input=day,model="test",embedding_model="test"),{},SimpleNamespace(calls=0,hits=0))
            self.assertTrue(result["integrity"]["passed"])
            self.assertFalse(formerly_unmatched.exists())
            self.assertEqual(len(list((Path(tmp)/"未匹配材料"/"文本").iterdir())),49)

    def test_broad_location_support_is_kept_without_claiming_precise_place(self):
        def source(sid,kind,place):
            return {"id":sid,"kind":kind,"platforms":[{"name":"AW-159反潜直升机","country":"韩国","quantity":2}],
                    "location":normalize_location(place),"mission":{"value":None},"start_time":None,"end_time":None,"movement":"活动"}
        news,opinion,radar=source("N001","news","东海油气田海域"),source("O001","opinion","东海"),source("R001","radar",None)
        radar.update({"sheet":"飞机","anomalies":[],"row_count":10})
        edge=lambda sid:{"candidate_id":sid,"platform_indices":[0],"relation":"same","confidence":.95,"reason":"同活动"}
        group={"opinion_id":"O001","radar_ids":["R001"],"radar_slots":{"0":"R001"},"alignment":{"opinion_edges":[edge("O001")],"radar_edges":[edge("R001")]}}
        facts=[{"platform_index":i,"field":field,"claim":claim,"supporting_sources":["N001"],"conflicting_sources":[],"status":"insufficient","reason":"精确地点仅一源"}
               for i,field,claim in [(0,"platform","AW-159"),(0,"country","韩国"),(0,"quantity","2架"),(-1,"location","东海油气田海域")]]
        verify={"same_event":True,"facts":facts,"unresolved_issues":[],"explanation":"支持宏观活动"}
        result=apply_guards(group,news,opinion,[radar],verify)
        self.assertEqual(result["status"],"可信事件")
        loc=next(f for f in result["verified_facts"] if f["field"]=="location")
        self.assertEqual(loc["claim"],"东海")
        detail=next(f for f in result["verification"]["facts"] if f["field"]=="location_detail")
        self.assertEqual(detail["status"],"insufficient")
        self.assertEqual(detail["claim"],"东海油气田海域")

        # A concrete location counter-value cannot be washed out by coarsening to a shared direction.
        import copy
        group=copy.deepcopy(group)
        original={"same_event":True,"facts":copy.deepcopy(facts[:4]),"unresolved_issues":[],"explanation":"有冲突"}
        original["facts"][3].update({"status":"conflict","conflicting_sources":["R001"]})
        radar["location"]=normalize_location("黄海")
        result=apply_guards(group,news,opinion,[radar],original)
        self.assertEqual(result["status"],"不可信事件")


if __name__=="__main__":
    unittest.main()
