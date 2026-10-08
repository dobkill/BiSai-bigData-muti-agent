"""Run several dates with bounded parallel inference and only two model switches."""
import argparse
from copy import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

from event_matcher.matching import build_index
from event_matcher.ollama_client import dump
from event_matcher.pipeline import prepare_day, extract_day, finish_day
from event_matcher.provenance import code_provenance


def run(args):
    if len(args.dates) != len(set(args.dates)):
        raise ValueError('Dates must be unique')
    for date in args.dates:
        if Path(date).name != date or date in {'.','..'}:
            raise ValueError('Each date must be a directory name')
        if not (args.input_root/date).is_dir():
            raise FileNotFoundError(args.input_root/date)
    if not args.cache_run:
        args.cache_run = 'parallel_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    records = args.output.resolve()/'运行记录'
    records.mkdir(parents=True, exist_ok=True)
    provenance = code_provenance()
    if provenance['current_code_dirty']:
        raise RuntimeError('Commit the matcher before running the full batch')
    inventory = [{'date':date,'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                 for date in args.dates for path in sorted((args.input_root/date).rglob('*')) if path.is_file()]
    inventory_path = records/'批次原始文件清单.json'
    if inventory_path.exists():
        previous = json.loads(inventory_path.read_text(encoding='utf-8'))
        expected = {(i['date'],i['path']):i['sha256'] for i in previous}
        for item in inventory:
            key = item['date'],item['path']
            if key in expected and expected[key] != item['sha256']:
                raise RuntimeError('Original input changed: '+item['path'])
        dump(records/(args.cache_run+'_原始文件清单.json'),inventory)
    else:
        dump(inventory_path,inventory)
    status_path = records/'批次状态.json'
    if status_path.exists():
        dump(records/('批次状态历史_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'),
             json.loads(status_path.read_text(encoding='utf-8')))
    status = {'started_at':datetime.now().astimezone().isoformat(),'dates':args.dates,'completed':[],'failed':[],
              'current':None,'state':'running','phase':'prepare','output':str(args.output.resolve()),
              'cache_run':args.cache_run,'workers':args.workers,'num_ctx':args.num_ctx,'ollama_url':args.ollama_url,
              'chat_url':args.chat_url,
              'code_provenance':provenance}
    started = time.monotonic()
    dump(status_path,status)

    def progress(phase,date=None):
        current = code_provenance()
        if current['code_sha256'] != provenance['code_sha256'] or current['git_commit'] != provenance['git_commit'] or current['current_code_dirty']:
            raise RuntimeError('Code changed during batch; refusing mixed-version exports')
        status.update(phase=phase,current=date)
        dump(status_path,status)
        print(f'[batch {phase}] {date or "all"} cache={args.cache_run} workers={args.workers}',flush=True)

    try:
        days=[]
        for date in args.dates:
            day_args=copy(args)
            day_args.input=args.input_root/date
            days.append(prepare_day(day_args))
        for day in days:
            progress('extract',day['args'].input.name)
            extract_day(day)
        # Keep embedding batches resident; never let different dates repeatedly evict the chat model.
        progress('embedding')
        days[0]['client'].unload(args.model)
        for day in days:
            progress('embedding',day['args'].input.name)
            day['index']=build_index(day['client'],day['texts']+day['radars'],day['output'])
        days[0]['client'].unload(args.embedding_model)
        for day in days:
            date=day['args'].input.name
            progress('match',date)
            manifest=finish_day(day,day['index'])
            if not manifest['integrity']['passed'] or not manifest['quality']['passed']:
                raise RuntimeError('Export validation failed: '+date)
            progress('export_checked',date)
            status['completed'].append(date)
            dump(status_path,status)
            print(f'[batch done] {date} trusted={manifest["trusted"]} untrusted={manifest["untrusted"]}',flush=True)
        unchanged=all(hashlib.sha256(Path(i['path']).read_bytes()).hexdigest()==i['sha256'] for i in inventory)
        if not unchanged:
            raise RuntimeError('Original input changed during inference')
        status.update(state='completed',phase='completed',current=None,
                      finished_at=datetime.now().astimezone().isoformat(),seconds=round(time.monotonic()-started,2),
                      original_hashes_unchanged=True)
        dump(status_path,status)
        print('[batch all complete] '+str(len(days))+' dates; original files unchanged',flush=True)
    except BaseException as error:
        status['failed'].append({'date':status['current'],'phase':status['phase'],'error':repr(error)})
        status.update(state='failed',seconds=round(time.monotonic()-started,2))
        dump(status_path,status)
        raise


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace',line_buffering=True)
    parser=argparse.ArgumentParser(description='分阶段批量匹配；同模型请求并发，分配和检查点按原顺序执行')
    parser.add_argument('--input-root',type=Path,default=Path('Datas/数据'))
    parser.add_argument('--output',type=Path,default=Path('Datas/匹配-初赛'))
    parser.add_argument('--dates',nargs='+',required=True)
    parser.add_argument('--model',default='qwen3.5:9b-q8_0')
    parser.add_argument('--embedding-model',default='bge-m3')
    parser.add_argument('--ollama-url',default='http://127.0.0.1:11435')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--num-ctx',type=int,default=12288)
    parser.add_argument('--chat-url',help='使用同一权重的llama-server执行并行文本推理')
    parser.add_argument('--cache-run',help='新名称从空缓存重推；同名称在本次缓存内恢复')
    run(parser.parse_args())


if __name__=='__main__':
    main()
