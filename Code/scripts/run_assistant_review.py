"""Replay explicitly saved in-session assistant decisions through real agents.

This does not call an API, Ollama, or another agent. Semantic choices must be
supplied in the plan; this script only binds source IDs, quotes and claim IDs.
The plan is locked to the complete source fingerprint. Outputs are isolated.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fusion_agent.agents.event_units import EventUnitAgent
from fusion_agent.agents.extraction import ExtractionAgent
from fusion_agent.agents.one_to_one import OneToOneAlignmentAgent
from fusion_agent.config import Paths, ReportSettings, Settings
from fusion_agent.ingestion import IngestionAgent
from fusion_agent.models import FACT_FIELDS
from fusion_agent.pipeline import AgentPipeline
from fusion_agent.utils import write_json


class SavedAssistantReview:
    def __init__(self, plan, docs, output, collect=False):
        self.plan, self.output, self.collect = plan, output, collect
        self.settings = SimpleNamespace(fail_on_error=not collect)
        self.documents = {d.id: d for d in docs}
        self.texts = {d.text: d for d in docs}
        self.records = []
        self.calls = self.cache_hits = 0  # Actual network requests, never replay counts.
        self.counts = Counter()

    @staticmethod
    def key(doc):
        return f"{doc.source}:{Path(doc.path).name[:3]}"

    def complete_json(self, system, user, schema):
        if schema['title'] == 'SemanticExtraction':
            doc = self.texts[user.split('原文：\n', 1)[1]]
            response = deepcopy(self.plan['extractions'][self.key(doc)])
            stage, request = 'extraction', {'document_id': doc.id, 'path': doc.path, 'text': doc.text}
        else:
            request = json.loads(user)
            news_id = request.get('news_id') or next(s['observation_id'] for s in request['sources'] if s['source'] == 'open_source')
            key = self.key(self.documents[news_id])
            stage = 'alignment' if schema['title'] == 'LinkReview' else 'verification'
            if self.collect and stage == 'alignment':
                self.records.append({'stage': stage, 'key': key, 'request': request})
                raise ValueError('Collecting candidates for in-session review; no decision supplied yet')
            decisions = self.plan['links'][key]
            if stage == 'alignment':
                response = {'decisions': [], 'radar_decisions': []}
                for candidate in request['opinion_candidates']:
                    opinion_key = self.key(self.documents[candidate['opinion_id']])
                    decision = decisions['opinions'][opinion_key]  # Missing review is an error.
                    response['decisions'].append({**decision, 'opinion_id': candidate['opinion_id'],
                                                  'news_quote': request['news_text'], 'opinion_quote': candidate['text']})
                for candidate in request['radar_candidates']:
                    decision = decisions['radars'][candidate['observation_id']]
                    response['radar_decisions'].append({**decision, 'observation_id': candidate['observation_id'],
                                                        'news_quote': request['news_text'], 'row_numbers': candidate['row_numbers']})
            else:
                chosen = self.plan['verifications'][key]
                response = {k: deepcopy(chosen[k]) for k in ('association', 'reason')}
                response['association_evidence'] = [
                    {'observation_id': s['observation_id'], 'quote': s['text'],
                     'row_numbers': sorted({n for row in s['radar_rows'] for n in row['row_numbers']})}
                    for s in request['sources']]
                response['fields'] = [
                    {'field': field, **deepcopy(chosen['fields'][field]),
                     'claim_ids': [c['claim_id'] for c in request['claims'] if c['field'] == field]}
                    for field in FACT_FIELDS]
                response['excluded_claims'] = []
        self.counts[stage] += 1
        self.records.append({'stage': stage, 'request': request, 'system': system, 'schema': schema,
                             'response': response, 'provider': 'assistant_review_replay'})
        return response

    def concurrency_snapshot(self):
        return {'mode': 'saved_assistant_decisions', 'network_requests': 0, 'replayed_responses': dict(self.counts)}

    def close(self):
        write_json(self.output / 'replayed_requests_responses.json', self.records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--collect-candidates', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    day = plan['date']
    data = root / 'Datas' / '数据'
    ingestion = IngestionAgent(data)
    if ingestion.fingerprint(day) != plan['source_fingerprint']:
        raise ValueError('Source changed; saved assistant decisions cannot be applied')
    out = args.output_dir.resolve()
    if not out.is_relative_to(root / 'Code' / 'outputs') or out == root / 'Code' / 'outputs':
        raise ValueError('Assistant tests must use an isolated directory under Code/outputs')
    settings = Settings(paths=Paths(data_dir=data, output_dir=out, cache_dir=out / '.cache', review_dir=out / 'review'),
                        report=ReportSettings(lookback_days=1))
    settings.extraction.mode = 'llm'
    settings.llm.concurrency = 1
    settings.llm.fail_on_error = True
    docs, batches, _ = ingestion.run(day)
    replay = SavedAssistantReview(plan, docs, out, args.collect_candidates)
    try:
        if args.collect_candidates:
            observations = ExtractionAgent(settings, replay).run(docs, batches)
            observations = EventUnitAgent().run(docs, observations, keep_documents=True)
            alignment = OneToOneAlignmentAgent(settings.alignment, replay)
            alignment.run(observations, docs)
            write_json(out / 'candidate_review_inputs.json', [r for r in replay.records if r['stage'] == 'alignment'])
            write_json(out / 'extracted_observations.json', [o.model_dump(mode='json') for o in observations])
            return
        pipeline = AgentPipeline(settings)  # LLM disabled: no HTTP client or service is initialized.
        pipeline.llm = replay
        for agent in (pipeline.extraction, pipeline.alignment, pipeline.verification):
            agent.llm = replay
        try:
            manifest = pipeline.run([day], force=True)
            manifest.update(model_provider='assistant_review_replay', model_name='current_conversation_assistant',
                            inference_mode='saved_in_session_review', decision_engine='assistant_review_replay',
                            actual_network_requests=0, replayed_responses=dict(replay.counts),
                            plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                            source_fingerprint=plan['source_fingerprint'],
                            explanation='当前对话模型阅读原文后保存的判断，经实际抽取、关联、核验及导出代码回放；无自动 API 推理。')
            write_json(out / 'manifest.json', manifest)
            progress = json.loads((out / 'progress.json').read_text(encoding='utf-8'))
            progress.update(model_provider='assistant_review_replay', model_name='current_conversation_assistant',
                            inference_mode='saved_in_session_review', actual_network_requests=0)
            write_json(out / 'progress.json', progress)
        finally:
            pipeline.close()
    finally:
        replay.close()


if __name__ == '__main__':
    main()
