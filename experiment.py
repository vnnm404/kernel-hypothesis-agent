"""Paired trials and append-only benchmark records; standard library only."""
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

CASES = ['attn_dense_hd128_' + s for s in ('causal', 'noncausal', 'gqa')]
ROOT = Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def events():
    p = Path('events.jsonl')
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []


def record(data):
    with open('events.jsonl', 'a') as f:
        f.write(json.dumps(data) + '\n')
        f.flush()
        os.fsync(f.fileno())


def capture(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def limited(cmd, seconds, **kwargs):
    """Kill the entire process group, including compiler/tool children, on timeout."""
    with subprocess.Popen(cmd, start_new_session=True, **kwargs) as p:
        try:
            return p.wait(timeout=seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return 124


def init():
    if not Path('run.json').exists():
        save('run.json', dict(schema_version=1, trial_id=Path.cwd().name,
             condition=os.environ.get('KHA_CONDITION', 'guided'),
             started_at=utc(), started_unix=time.time(), started_monotonic=time.monotonic(), pair=None, position=None))


def measure(source):
    # Serialize all benchmark invocations, including accidental parallel tool calls.
    with open('.benchmark.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        meta = read('run.json')
        phase = os.environ.get('KHA_PHASE', 'search')
        starts = [e for e in events() if e['event'] == 'benchmark_start']
        if phase == 'search' and sum(e['phase'] == phase for e in starts) >= int(os.environ.get('BENCHMARKS', '12')):
            record(dict(event='budget_exhausted', timestamp=utc(), phase=phase))
            print('Search benchmark budget exhausted. Write result.md and stop.', flush=True)
            return 2
        mid = f'{len(starts)+1:04d}'
        folder = Path('measurements') / mid
        folder.mkdir(parents=True)
        started = time.monotonic()
        common = dict(schema_version=1, trial_id=meta['trial_id'], pair=meta['pair'],
                      condition=meta['condition'], measurement_id=mid, phase=phase,
                      started_at=utc(), elapsed_s=started-meta['started_monotonic'],
                      agent_elapsed_s=(started-meta['agent_started_monotonic']) if 'agent_started_unix' in meta else None,
                      kernel_sha256=None, snapshot=str(folder/'kernel.cu'), log=str(folder/'benchmark.log'))
        try:
            shutil.copyfile(source, folder/'kernel.cu')
            common['kernel_sha256'] = sha(folder/'kernel.cu')
        except OSError as e:
            record(dict(common, event='benchmark_start'))
            end = dict(common, event='benchmark_end', finished_at=utc(), duration_s=time.monotonic()-started,
                       returncode=1, valid=False, cases=[], error=str(e))
            record(end)
            save(folder/'measurement.json', end)
            Path(common['log']).write_text(str(e)+'\n')
            print(e, file=sys.stderr)
            return 1
        record(dict(common, event='benchmark_start'))
        if Path('notes.md').exists():
            shutil.copyfile('notes.md', folder/'notes.md')
        print(f'Measurement {mid} ({phase}), kernel {common["kernel_sha256"]}', flush=True)
        # Stable build path reuses compilation for repeated measurements of identical bytes.
        Path('.evaluation').mkdir(exist_ok=True)
        shutil.copyfile(folder/'kernel.cu', '.evaluation/candidate.cu')
        cmd = [sys.executable, '-m', 'bench.eval', '--target', 'blackwell', '--cand',
               str(Path('.evaluation/candidate.cu').resolve()), '--namespace', 'FA3_LIB_NAME',
               '--gpu', os.environ.get('GPU', '0'), '--set', 'all', '--no-cudnn', '--no-flashinfer',
               '--configs', ','.join(CASES)]
        with open(common['log'], 'w') as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        text = Path(common['log']).read_text(errors='replace')
        rows = []
        for name in CASES:
            match = re.search(r'^\s+OK\s+' + name + r'\s+.*?\(FA4\s+([\d.]+)us / ours\s+([\d.]+)us\)', text, re.M)
            ref, cand = map(float, match.groups()) if match else (None, None)
            rows.append(dict(case=name, correct=bool(match) if re.search(r'^\s+\S+\s+'+name+r'\s', text, re.M) else None, candidate_us=cand, reference_us=ref,
                             speedup_vs_fa4=ref/cand if cand and ref else None))
        valid = (rc == 0 and all(r['candidate_us'] and r['reference_us'] for r in rows)
                 and bool(re.search(r'^correct: 3/3 +unsupported: 0 +wrong: 0', text, re.M)))
        build = re.search(r'\[build\] ok in ([\d.]+)s', text)
        end = dict(common, event='benchmark_end', finished_at=utc(), duration_s=time.monotonic()-started,
                   build_s=float(build[1]) if build else None, returncode=rc, valid=bool(valid), cases=rows)
        record(end)
        save(folder/'measurement.json', end)
        print(text, end='', flush=True)
        if not valid:
            print('Incomplete/failed benchmark: ' + common['log'], file=sys.stderr)
        return 0 if valid else 1


def agent(args):
    meta = read('run.json')
    meta.update(agent_started_at=utc(), agent_started_unix=time.time(), agent_started_monotonic=time.monotonic())
    save('run.json', meta)
    with open('instructions.md') as prompt, open('trace.jsonl', 'w') as trace, open('agent.stderr', 'w') as err:
        rc = limited(['claude', '-p', '--verbose', '--output-format', 'stream-json',
                      '--no-session-persistence', '--safe-mode', '--tools',
                      'Read,Write' if os.environ.get('KHA_SMOKE') == '1' else 'Read,Write,Edit,Bash,Glob,Grep',
                      '--allowedTools',
                      'Read,Write' if os.environ.get('KHA_SMOKE') == '1' else 'Read,Write,Edit,Bash,Glob,Grep',
                      *args], int(os.environ.get('SECONDS_PER_RUN', '3600')),
                     stdin=prompt, stdout=trace, stderr=err)
    meta.update(agent_finished_at=utc(), agent_duration_s=time.monotonic()-meta['agent_started_monotonic'], agent_exit_code=rc)
    save('run.json', meta)
    return rc


def summarize(folder, rc):
    meta = read(folder/'run.json')
    ev = [json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()] if (folder/'events.jsonl').exists() else []
    ends = [e for e in ev if e['event'] == 'benchmark_end']
    # A killed compiler cannot write its own completion record. Preserve that attempt.
    completed_ids = {e['measurement_id'] for e in ends}
    for start in [e for e in ev if e['event'] == 'benchmark_start' and e['measurement_id'] not in completed_ids]:
        end = dict(start, event='benchmark_end', finished_at=utc(), duration_s=None,
                   returncode=None, valid=False, cases=[], error='interrupted: no completion record')
        with open(folder/'events.jsonl', 'a') as f:
            f.write(json.dumps(end)+'\n')
        save(folder/'measurements'/start['measurement_id']/'measurement.json', end)
        ends.append(end)
    medians = {}
    repeats = int(os.environ.get('REPEATS', '3'))
    for phase in ('seed', 'final'):
        es = [e for e in ends if e['phase'] == phase]
        medians[phase] = {n: statistics.median([r['candidate_us'] for e in es for r in e['cases'] if r['case'] == n])
                         for n in CASES} if len(es) == repeats and all(e['valid'] for e in es) else None
    gain = None
    regress = None
    if all(medians.values()):
        gain = math.prod(medians['seed'][n]/medians['final'][n] for n in CASES)**(1/len(CASES))
        regress = max(medians['final'][n]/medians['seed'][n] for n in CASES)
    trace = []
    if (folder/'trace.jsonl').exists():
        for line in (folder/'trace.jsonl').read_text().splitlines():
            try:
                trace.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Preserve partial output from a killed process.
    result = next((e for e in reversed(trace) if e.get('type') == 'result'), {})
    init_event = next((e for e in trace if e.get('type') == 'system' and e.get('subtype') == 'init'), {})
    out = dict(meta, finished_at=utc(), exit_code=rc,
               status='completed' if rc == 0 else 'failed',
               search_benchmarks=sum(e['event'] == 'benchmark_start' and e['phase'] == 'search' for e in ev),
               valid_search_benchmarks=sum(e['phase'] == 'search' and e['valid'] for e in ends),
               seed_median_us=medians['seed'], final_median_us=medians['final'],
               final_speedup_vs_seed=gain, worst_latency_ratio_vs_seed=regress,
               qualifies=bool(rc == 0 and gain and gain >= 1.02 and regress <= 1.03),
               actual_model=init_event.get('model'), num_turns=result.get('num_turns'),
               usage=result.get('usage'), model_usage=result.get('modelUsage'),
               reported_cost_usd=result.get('total_cost_usd'), agent_result_error=result.get('is_error'))
    save(folder/'summary.json', out)
    return out


def run(flash, args):
    config = {name: int(os.environ.get(name, default)) for name, default in
              [('PAIRS', '6'), ('BENCHMARKS', '12'), ('REPEATS', '3'), ('SECONDS_PER_RUN', '3600'), ('ORDER_SEED', '0')]}
    if any(config[k] < 1 for k in config if k != 'ORDER_SEED'):
        raise SystemExit('PAIRS, BENCHMARKS, REPEATS and SECONDS_PER_RUN must be positive.')
    # These options alter the experiment itself rather than simply selecting the model.
    rest = list(args)
    while rest:
        flag = rest.pop(0)
        if flag in ('--model', '--effort') and rest:
            rest.pop(0)
        elif not flag.startswith(('--model=', '--effort=')):
            raise SystemExit('Comparison mode accepts only Claude --model and --effort options.')
    os.environ.update({k: str(v) for k, v in config.items()})
    batch = Path(tempfile.mkdtemp(prefix='experiment-', dir=ROOT/'runs'))
    first = ['baseline', 'guided'] * (config['PAIRS']//2) + (['baseline'] if config['PAIRS'] % 2 else [])
    random.Random(config['ORDER_SEED']).shuffle(first)
    schedule = [(pair, condition, position) for pair, lead in enumerate(first, 1)
                for position, condition in enumerate((lead, 'guided' if lead == 'baseline' else 'baseline'), 1)]
    manifest = dict(schema_version=1, started_at=utc(), config=config, cases=CASES, schedule=schedule,
                    seed_sha256=sha(Path(flash)/'kernels/fa4_blackwell.cu'),
                    flash_commit=capture(['git', '-C', flash, 'rev-parse', 'HEAD']),
                    wrapper_commit=capture(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                    claude_version=capture(['claude', '--version']), claude_options=args,
                    runner_sha256={n: sha(ROOT/n) for n in ('run.sh', 'experiment.py')},
                    benchmark_sha256={n: sha(Path(flash)/'bench'/n) for n in ('__init__.py', 'eval.py', 'scenarios.py')},
                    gpu_index=int(os.environ.get('GPU', '0')), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    gpu=capture(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader']),
                    nvcc=capture(['nvcc', '--version']), python=sys.version,
                    packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()})
    save(batch/'experiment.json', manifest)
    print('Dataset: ' + str(batch), flush=True)
    summaries = []
    for pair, condition, position in schedule:
        trial = batch/f'pair-{pair:02d}-{condition}'
        trial.mkdir()
        save(trial/'run.json', dict(schema_version=1, trial_id=trial.name, pair=pair, condition=condition,
             position=position, started_at=utc(), started_unix=time.time(), started_monotonic=time.monotonic(), seed_sha256=manifest['seed_sha256']))
        env = dict(os.environ, KHA_TRIAL_DIR=str(trial), KHA_CONDITION=condition, PYTHON_BIN=sys.executable)
        print(f'Pair {pair}/{config["PAIRS"]}: {condition}', flush=True)
        with open(trial/'runner.log', 'w') as log:
            rc = subprocess.call(['bash', str(ROOT/'run.sh'), flash, *args], env=env, stdout=log, stderr=subprocess.STDOUT)
        summaries.append(summarize(trial, rc))
        # Update after every trial, including failures, so partial batches remain analyzable.
        columns = ['trial_id', 'pair', 'condition', 'position', 'status', 'exit_code', 'actual_model',
                   'search_benchmarks', 'valid_search_benchmarks', 'final_speedup_vs_seed',
                   'worst_latency_ratio_vs_seed', 'qualifies', 'agent_duration_s', 'num_turns', 'reported_cost_usd']
        with open(batch/'summary.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summaries)
        with open(batch/'measurements.jsonl', 'w') as f:
            for directory in sorted(batch.glob('pair-*')):
                if (directory/'events.jsonl').exists():
                    for line in (directory/'events.jsonl').read_text().splitlines():
                        if json.loads(line)['event'] == 'benchmark_end':
                            f.write(line+'\n')
        print(f'{trial.name}: {summaries[-1]["status"]}; results saved', flush=True)
    manifest['finished_at'] = utc()
    save(batch/'experiment.json', manifest)
    print('Dataset complete: ' + str(batch), flush=True)
    return int(any(s['exit_code'] for s in summaries))


if __name__ == '__main__':
    command, *args = sys.argv[1:]
    if command == 'run':
        sys.exit(run(args[0], args[1:]))
    if command == 'init':
        init()
    elif command == 'measure':
        sys.exit(measure(args[0] if args else 'candidate.cu'))
    elif command == 'agent':
        sys.exit(agent(args))
    elif command == 'benchmark':
        # Parent owns the timeout; evaluator and its compiler children share this group.
        sys.exit(limited([sys.executable, __file__, 'measure', *args], 900))
