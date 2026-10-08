"""Measure real extraction with full evidence validation and no answer-cache reuse."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from event_matcher.agents import extract_sources
from event_matcher.data import load_texts
from event_matcher.ollama_client import OllamaClient, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--chat-url')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--worker-counts',type=int,nargs='+',help='Compare specified concurrency settings on identical materials')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = load_texts(args.input)[:args.samples*4]
    if len(sources) != args.samples*4:
        raise ValueError('Not enough real prompts for benchmark')

    report = {'chat_url':args.chat_url,'workers':args.workers,'samples':args.samples,'context':12288,'measurements':[]}
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    def client(workers,label):
        return OllamaClient(args.output.parent/(args.output.stem+'_'+stamp)/label,
                            'qwen3.5:9b-q8_0','bge-m3',args.ollama_url,workers=workers,chat_url=args.chat_url)
    warm_client=client(1,'warmup')
    warm_client.wait_ready()
    extract_sources(warm_client,sources[:4],lambda data:None)
    for workers in dict.fromkeys(args.worker_counts or [1,args.workers]):
        started = time.monotonic()
        run_client=client(workers,str(workers))
        outputs=extract_sources(run_client,sources,lambda data:None)
        seconds = time.monotonic()-started
        logs=[json.loads(line) for line in (run_client.cache/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        measurement = {'workers':workers,'seconds':seconds,'total_output_tokens':sum(r['eval_count'] or 0 for r in logs),
                       'requests_per_minute':args.samples*60/seconds,'valid_documents':len(outputs),
                       'http_calls':run_client.calls,'cache_hits':run_client.hits,'max_active_http_requests':run_client.max_active_requests}
        report['measurements'].append(measurement)
        print(json.dumps(measurement,ensure_ascii=False),flush=True)
        dump(args.output,report)
    report['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    if len(report['measurements']) > 1:
        report['speedup'] = report['measurements'][0]['seconds']/report['measurements'][-1]['seconds']
    dump(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='measurements'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
