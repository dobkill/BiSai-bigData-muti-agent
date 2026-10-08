import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from event_matcher.concurrency import ordered_map
from event_matcher.models import Alignment
from event_matcher.ollama_client import OllamaClient, dump


class ConcurrencyTests(unittest.TestCase):
    def test_parallel_completion_does_not_change_input_order(self):
        barrier=threading.Barrier(4)
        def work(item):
            barrier.wait(timeout=10)
            return item*2
        self.assertEqual(list(ordered_map(SimpleNamespace(workers=4),work,range(12))),list(range(0,24,2)))

    def test_failure_stops_submission_and_surfaces_the_original_error(self):
        seen=[]
        def work(item):
            seen.append(item)
            if item==0:
                raise ValueError('broken evidence')
            return item
        with self.assertRaisesRegex(ValueError,'broken evidence'):
            list(ordered_map(SimpleNamespace(workers=2),work,range(100)))
        self.assertTrue(set(seen)<={0,1})

    def test_slow_first_request_does_not_leave_other_slots_idle(self):
        replenished=threading.Event()
        def work(item):
            if item==0 and not replenished.wait(timeout=5):
                raise RuntimeError('Later work was blocked by output ordering')
            if item==2:
                replenished.set()
            return item
        self.assertEqual(list(ordered_map(SimpleNamespace(workers=2),work,range(6))),list(range(6)))

    def test_thread_local_sessions_and_complete_concurrent_request_logs(self):
        barrier=threading.Barrier(4)
        sessions=[]
        class Session:
            def __init__(self):
                sessions.append(self)
                self.trust_env=True
            def post(self,url,json,timeout):
                barrier.wait(timeout=10)
                return SimpleNamespace(status_code=200,raise_for_status=lambda:None,
                                       json=lambda:{'eval_count':json['index'],'done_reason':'stop'})
        with tempfile.TemporaryDirectory() as tmp,patch('event_matcher.ollama_client.requests.Session',Session):
            client=OllamaClient(Path(tmp),'test','embed','http://test',workers=4)
            answers=list(ordered_map(client,lambda i:client.request('/test',{'model':'test','index':i}),range(8)))
            self.assertEqual([r['eval_count'] for r in answers],list(range(8)))
            records=[json.loads(line) for line in (Path(tmp)/'requests.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),8)
            self.assertEqual(client.calls,8)
            self.assertEqual(client.active_requests,0)
            self.assertEqual(client.max_active_requests,4)
            self.assertEqual(len(sessions),4)
            self.assertTrue(all(not s.trust_env for s in sessions))

    def test_atomic_writes_can_share_destination_without_corrupting_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.json'
            list(ordered_map(SimpleNamespace(workers=8),lambda i:dump(path,{'id':i,'payload':'x'*10000}),range(32)))
            self.assertIn(json.loads(path.read_text())['id'],range(32))
            self.assertEqual(list(Path(tmp).glob('*.tmp')),[])

    def test_llama_adapter_preserves_schema_evidence_and_full_raw_response(self):
        sent=[]
        original={'choices':[{'message':{'role':'assistant','content':'{}'},'finish_reason':'length'}],
                  'usage':{'completion_tokens':17,'prompt_tokens':41}}
        class Session:
            def post(self,url,json,timeout):
                sent.append((url,json))
                return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:original)
        with tempfile.TemporaryDirectory() as tmp,patch('event_matcher.ollama_client.requests.Session',Session):
            client=OllamaClient(Path(tmp),'qwen3.5:9b-q8_0','embed','http://ollama',chat_url='http://llama')
            body={'model':client.model,'messages':[{'role':'user','content':'原始证据'}],'format':{'type':'object'},
                  'options':{'temperature':0,'num_predict':5000,'seed':42}}
            response=client.request('/api/chat',body)
            self.assertEqual(sent[0][0],'http://llama/v1/chat/completions')
            self.assertEqual(sent[0][1]['messages'],body['messages'])
            self.assertEqual(sent[0][1]['response_format']['json_schema']['schema'],body['format'])
            self.assertEqual(response['done_reason'],'length')
            self.assertEqual(response['raw_backend_response'],original)
            self.assertEqual(sent[0][1]['chat_template_kwargs'],{'enable_thinking':False})

    def test_backend_context_and_cache_namespace_prevent_old_answer_reuse(self):
        calls=[]
        def request(endpoint,body):
            calls.append(body)
            return {'message':{'content':'{"opinion_edges":[],"radar_edges":[],"ambiguity":""}'}}
        with tempfile.TemporaryDirectory() as tmp:
            for namespace,chat_url,context in [('old',None,12288),('old','http://llama',12288),
                                               ('old','http://llama',16384),('fresh','http://llama',12288)]:
                client=OllamaClient(Path(tmp)/namespace,'test','embed','http://ollama',num_ctx=context,chat_url=chat_url)
                client.request=request
                client.chat('alignment','instructions',{'news':'evidence'},Alignment)
            self.assertEqual(len(calls),4)
            client.chat('alignment','instructions',{'news':'evidence'},Alignment)
            self.assertEqual(len(calls),4)

    def test_server_readiness_rejects_short_context_and_false_concurrency(self):
        for slots,context in [(1,12288),(8,4096)]:
            with self.subTest(slots=slots,context=context),tempfile.TemporaryDirectory() as tmp:
                class Session:
                    def get(self,url,timeout):
                        payload={'total_slots':slots,'default_generation_settings':{'n_ctx':context}}
                        return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:payload)
                with patch('event_matcher.ollama_client.requests.Session',Session):
                    client=OllamaClient(Path(tmp),'test','embed','http://ollama',workers=8,chat_url='http://llama')
                    with self.assertRaises(RuntimeError):
                        client.wait_ready(timeout=1)


if __name__=='__main__':
    unittest.main()
