"""Run independent inference in parallel, consuming outputs in the original order."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice


def ordered_map(client, function, items):
    workers = getattr(client, 'workers', 1)
    if workers == 1:
        yield from map(function, items)
        return
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='inference')
    iterator = iter(enumerate(items))
    pending = {executor.submit(function,item):index for index,item in islice(iterator,workers)}
    ready = {}
    next_output = 0
    try:
        while pending:
            completed,_ = wait(pending,return_when=FIRST_COMPLETED)
            # Detect a failed task before scheduling more inference.
            for future in completed:
                if future.exception() is not None:
                    future.result()
            for future in completed:
                ready[pending.pop(future)] = future.result()
                next_item = next(iterator,None)
                if next_item is not None:
                    index,item = next_item
                    pending[executor.submit(function,item)] = index
            # Refill on every completion; a slow first request cannot idle other slots.
            while next_output in ready:
                yield ready.pop(next_output)
                next_output += 1
    finally:
        executor.shutdown(wait=True,cancel_futures=True)
