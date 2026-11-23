"""Small shared audit primitives. Importing this module never imports torch."""
import contextlib
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def event(kind, **fields):
    row = dict(utc=utc(), monotonic=time.perf_counter(), pid=os.getpid(),
               stage=os.environ.get('CURATED_STAGE'), kind=kind, **fields)
    root = os.environ.get('CURATED_RUN_ROOT')
    if root:
        path = Path(root) / 'evidence/events.jsonl'
        # Driver and worker write one small O_APPEND record per event.
        with path.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
            stream.flush()
    print('CURATED_EVENT ' + json.dumps(row, ensure_ascii=False, default=str), flush=True)
    return row


def sync_cuda():
    import sys
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()


@contextlib.contextmanager
def timed(kind, **fields):
    sync_cuda()
    start = time.perf_counter()
    event(kind + '_start', **fields)
    try:
        yield
        sync_cuda()
    except BaseException:
        event(kind + '_failed', elapsed_seconds=time.perf_counter() - start,
              traceback=traceback.format_exc(), **fields)
        raise
    else:
        event(kind + '_done', elapsed_seconds=time.perf_counter() - start, **fields)


def trace_load(fn, *args, **kwargs):
    """Called at actual loader call sites, never a fabricated top-level counter."""
    root = os.environ.get('CURATED_RUN_ROOT')
    if root and not (Path(root) / 'evidence/BATCH_STARTED.json').exists():
        row = event('batch_start', boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        save_json(Path(root) / 'evidence/BATCH_STARTED.json', row)
    owner = getattr(fn, '__self__', None)
    owner_name = getattr(owner, '__name__', type(owner).__name__) if owner is not None else ''
    path = args[0] if args and isinstance(args[0], (str, Path)) else kwargs.get('checkpoint', kwargs.get('pretrained'))
    if getattr(fn, '__name__', '') == 'load_checkpoint' and len(args) > 1:
        path = args[1]
    details = dict(loader=f'{getattr(fn, "__module__", "")}.{owner_name}.{getattr(fn, "__qualname__", str(fn))}',
                   requested_path=str(path) if path is not None else None,
                   subfolder=kwargs.get('subfolder'))
    if path is not None and Path(str(path)).exists():
        details['resolved_path'] = str(Path(str(path)).resolve())
    elif isinstance(path, str) and '/' in path and not path.startswith('/'):
        try:
            from huggingface_hub import snapshot_download
            details['resolved_cache_snapshot'] = snapshot_download(path, local_files_only=True)
        except Exception as exc:
            details['cache_resolution_note'] = type(exc).__name__
    if getattr(fn, '__name__', '') == 'create_model_and_transforms':
        import open_clip
        from huggingface_hub import hf_hub_download
        cfg = open_clip.get_pretrained_cfg(kwargs['model_name'], kwargs['pretrained'])
        hub = cfg.get('hf_hub', '')
        if hub:
            repo_id, filename = hub.rsplit('/', 1)
            filename = filename or 'open_clip_pytorch_model.bin'
            details['resolved_cache_file'] = hf_hub_download(repo_id, filename, local_files_only=True)
        details['model_name'] = kwargs.get('model_name')
    with timed('model_load', **details):
        return fn(*args, **kwargs)


def artifact_record(root, paths):
    root = Path(root).resolve()
    rows = []
    for path in sorted(set(Path(p).resolve() for p in paths)):
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Invalid/empty artifact: {path}')
        rows.append(dict(path=str(path.relative_to(root)), sha256=digest(path), bytes=path.stat().st_size))
    if not rows:
        raise ValueError('No artifacts produced')
    return rows


def verify_records(root, rows):
    for row in rows:
        path = Path(root) / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'Artifact changed/missing: {path}')


def summarize(root):
    root = Path(root)
    events = [json.loads(line) for line in (root / 'evidence/events.jsonl').read_text().splitlines()]
    rows = [e for e in events if 'elapsed_seconds' in e]
    # Shared load and sample processing are separate: nested model_load rows not added twice.
    successes = [e for e in rows if e['kind'] == 'sample_done']
    starts = [e for e in events if e['kind'] == 'batch_start']
    ends = [e for e in events if e['kind'] == 'batch_done']
    model_loads = [e for e in events if e['kind'] == 'model_load_done']
    counts = {}
    for e in model_loads:
        key = f"{e['stage']}|{e['loader']}|{e.get('requested_path')}|{e.get('subfolder')}"
        counts[key] = counts.get(key, 0) + 1
    report = dict(events=events, timed_operations=rows, actual_successful_load_counts=counts,
                  sample_processing_seconds={sid: sum(e['elapsed_seconds'] for e in successes if e.get('sample_id') == sid)
                                             for sid in sorted({e['sample_id'] for e in successes})},
                  successful_sample_stage_seconds=sum(e['elapsed_seconds'] for e in successes),
                  shared_stage_load_seconds=sum(e['elapsed_seconds'] for e in rows if e['kind'] == 'shared_load_done'),
                  batch_first_start=starts[0]['utc'] if starts else None,
                  batch_completed_utc=ends[-1]['utc'] if ends else None,
                  batch_wall_seconds_including_interruptions=(dt.datetime.fromisoformat(ends[-1]['utc']) - dt.datetime.fromisoformat(starts[0]['utc'])).total_seconds() if starts and ends else None,
                  completed_attempt_monotonic_seconds=ends[-1]['elapsed_seconds'] if ends else None,
                  note='Sample sums are processing only, not independent end-to-end times. Shared loads counted once. UTC wall span includes failures/interruption; attempt elapsed uses monotonic.')
    save_json(root / 'timings.json', report)
    with (root / 'timings.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['utc','kind','stage','sample_id','elapsed_seconds','pid'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return report
