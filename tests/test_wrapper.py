"""No GPU/model calls: check shell orchestration against local stand-ins."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PAT = 'fake-snowflake-pat'
SNOWFLAKE = 'https://snowhouse.snowflakecomputing.com/api/v2/cortex/anthropic'


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='kernel wrapper ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('run.sh', 'experiment.py'):
            shutil.copy2(ROOT / name, self.root / name)
        self.flash = self.root / 'existing flash'
        (self.flash / 'bench').mkdir(parents=True)
        (self.flash / 'kernels').mkdir()
        (self.flash / 'kernels/fa4_blackwell.cu').write_text('original kernel')
        (self.flash / 'torch.py').write_text('''from types import SimpleNamespace
cuda = SimpleNamespace(set_device=lambda i: None, get_device_capability=lambda: (10,0), get_device_name=lambda: 'fixture')
''')
        (self.flash / 'bench/__init__.py').write_text('')
        (self.flash / 'bench/scenarios.py').write_text('''from types import SimpleNamespace
def get_configs(name):
 return [SimpleNamespace(name='attn_dense_hd128_'+s) for s in ['causal','noncausal','gqa']]
''')
        (self.flash / 'bench/eval.py').write_text('''import json, os, sys
TARGETS = {'blackwell': 'fixture'}
def baseline_available(target): return not os.environ.get('BAD_REFERENCE')
def baseline_import_error(target): return 'missing reference fixture'
if __name__ == '__main__':
 with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(sys.argv)+'\\n')
 if os.environ.get('BAD_BENCH'):
  print('correct: 2/3   unsupported: 1   wrong: 0')
  print('perf over 2 configs: geomean=1.0x')
 else:
  from pathlib import Path
  if os.environ.get('SLOW_SEARCH') and not os.environ.get('KHA_PHASE'):
   import time; time.sleep(60)
  latency = 80.0 if Path(sys.argv[sys.argv.index('--cand')+1]).read_text() == 'changed kernel' else 100.0
  for suffix in ('causal', 'noncausal', 'gqa'):
   print('  OK attn_dense_hd128_'+suffix+f' 1.000x (FA4 100.0us / ours {latency:.1f}us)')
  print('correct: 3/3   unsupported: 0   wrong: 0')
  print('perf over 3 configs: geomean=1.0x')
''')
        bins = self.root / 'bin'
        bins.mkdir()
        (bins / 'nvcc').write_text('#!/bin/sh\nexit 0\n')
        claude = bins / 'claude'
        claude.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
if '--help' in sys.argv:
 print('--safe-mode --no-session-persistence'); sys.exit(0)
if '--version' in sys.argv:
 print('fixture-claude'); sys.exit(0)
Path(os.environ['CLAUDE_LOG']).write_text(json.dumps({'argv': sys.argv, 'stdin': sys.stdin.read(),
 'key': os.environ.get('ANTHROPIC_API_KEY'), 'url': os.environ.get('ANTHROPIC_BASE_URL')}))
print(json.dumps({'type':'system', 'subtype':'init', 'model':'fixture-model'}))
if os.environ.get('AGENT_SLEEP'):
 import time; time.sleep(60)
for i in range(int(os.environ.get('SEARCH_COUNT', '0'))):
 import subprocess
 Path('candidate.cu').write_text('changed kernel')
 Path('notes.md').write_text(f'Attempt {i+1}')
 r = subprocess.run(['bash', './benchmark.sh'], capture_output=True, text=True)
 print(json.dumps({'type':'user', 'message': {'content': r.stdout + r.stderr}}))
if not os.environ.get('NO_RESULT'):
 Path('result.md').write_text('Fixture result; no real GPU or model used.')
if os.environ.get('CHANGE_BEST'):
 Path('best.cu').write_text('changed kernel')
print(json.dumps({'type':'result', 'is_error': bool(os.environ.get('AGENT_ERROR'))}))
''')
        for p in bins.iterdir():
            p.chmod(0o755)
        # Fake credentials only: drop inherited auth and use a fixture PAT path.
        self.pat = self.root / 'fake-pat'
        self.pat.write_text(PAT + '\n')
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('ANTHROPIC_', 'CLAUDE_', 'OPENAI_'))}
        self.env = env | {'PATH': f'{bins}:/usr/bin:/bin', 'PYTHON_BIN': sys.executable, 'SNOWFLAKE_PAT_FILE': str(self.pat),
                          'KHA_TRIAL_DIR': str(self.root / 'runs/single'), 'REPEATS': '1',
                          'PAIRS': '2', 'BENCHMARKS': '2', 'SECONDS_PER_RUN': '10',
                          'CALL_LOG': str(self.root / 'calls'), 'CLAUDE_LOG': str(self.root / 'claude-call')}

    def run_wrapper(self, *args, **env):
        return subprocess.run(['bash', str(self.root / 'run.sh'), *args],
                              env=self.env | env, capture_output=True, text=True, timeout=20)

    def calls(self):
        return [json.loads(s) for s in (self.root / 'calls').read_text().splitlines()]

    def claude_call(self):
        return json.loads((self.root / 'claude-call').read_text())

    def mock_downloads(self):
        """Exercise first-run setup without network traffic or CUDA packages."""
        bins = self.root / 'bin'
        (bins / 'nvcc').write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0'\n")
        (bins / 'g++').write_text('#!/bin/sh\nexit 0\n')
        uv = self.root / 'fake-uv'
        uv.write_text(f'#!{sys.executable}\n' + '''import json, os, sys, shlex
from pathlib import Path
with open(os.environ['SETUP_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('FAIL_INSTALL') and sys.argv[1] == 'pip': sys.exit(1)
if sys.argv[1] == 'venv':
 p = Path(sys.argv[-1]) / 'bin/python'
 p.parent.mkdir(parents=True, exist_ok=True)
 p.write_text('#!/bin/sh\\nexec '+shlex.quote(sys.executable)+' "$@"\\n')
 p.chmod(0o755)
''')
        uv.chmod(0o755)
        (bins / 'curl').write_text(f'#!{sys.executable}\n' + '''import sys
from pathlib import Path
Path(sys.argv[sys.argv.index('-o')+1]).write_text('#!/bin/sh\\nmkdir -p "$UV_UNMANAGED_INSTALL"\\ncp "$FAKE_UV" "$UV_UNMANAGED_INSTALL/uv"\\n')
''')
        (bins / 'git').write_text(f'#!{sys.executable}\n' + '''import os, shutil, sys
from pathlib import Path
a = sys.argv[1:]
if a[0] == 'init':
 (Path(a[-1]) / '.git').mkdir(parents=True)
elif a[0] == '-C':
 dest, op = Path(a[1]), a[2]
 if op == 'rev-parse':
  if not (dest / '.git/revision').exists(): sys.exit(1)
  print((dest / '.git/revision').read_text())
 elif op == 'fetch': (dest / '.git/revision').write_text(a[-1])
 elif op == 'checkout' and dest.name == 'flash-cuda':
  shutil.copytree(os.environ['FIXTURE_FLASH'], dest, dirs_exist_ok=True)
''')
        for p in bins.iterdir():
            p.chmod(0o755)
        self.env.update(FAKE_UV=str(uv), SETUP_LOG=str(self.root / 'setup-log'),
                        FIXTURE_FLASH=str(self.flash), PYTHONPATH=str(self.flash))

    def test_no_path_sets_up_dependencies_and_reuses_environment(self):
        self.mock_downloads()
        for _ in range(2):
            r = self.run_wrapper('--smoke')
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = [json.loads(s) for s in (self.root / 'setup-log').read_text().splitlines()]
        self.assertEqual(sum(c[0] == 'venv' for c in calls), 1)
        self.assertTrue(any('torch==2.9.1' in c for c in calls))
        self.assertTrue(any('--no-deps' in c and 'quack-kernels==0.5.3' in c for c in calls))
        self.assertEqual(len(self.calls()), 4)
        self.assertNoSecret(r, PAT)

    def test_setup_failure_stops_before_gpu_and_can_be_retried(self):
        self.mock_downloads()
        r = self.run_wrapper('--smoke', FAIL_INSTALL='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.root / 'calls').exists())
        self.assertFalse((self.root / 'claude-call').exists())
        r = self.run_wrapper('--smoke')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_missing_pat_with_no_path_stops_before_downloads(self):
        self.mock_downloads()
        self.pat.unlink()
        r = self.run_wrapper()
        self.assertStoppedBeforeGpu(r, 'needs your PAT')
        self.assertFalse((self.root / '.deps').exists())

    def assertNoSecret(self, r, *secrets):
        runs = [p.read_text() for p in (self.root / 'runs').rglob('*') if p.is_file()]
        text = r.stdout + r.stderr + ''.join(runs)
        for secret in secrets:
            self.assertNotIn(secret, text)

    def assertStoppedBeforeGpu(self, r, message):
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(message, r.stderr)
        self.assertNotIn('Ready', r.stdout)
        self.assertFalse((self.root / 'calls').exists())
        self.assertFalse((self.root / 'claude-call').exists())

    def test_full_run_paths_with_spaces_and_existing_checkout_untouched(self):
        before = {str(p): p.read_bytes() for p in self.flash.rglob('*') if p.is_file()}
        r = self.run_wrapper(str(self.flash), '--model', 'configured-model', CHANGE_BEST='1')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        after = {str(p): p.read_bytes() for p in self.flash.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][calls[0].index('--namespace') + 1], 'FA3_LIB_NAME')
        self.assertEqual((self.root / 'runs/single/measurements/0002/kernel.cu').read_text(), 'changed kernel')
        argv = self.claude_call()['argv']
        self.assertIn('configured-model', argv)
        self.assertNotIn('claude-opus-5[1m]', argv)
        self.assertIn('smallest test', self.claude_call()['stdin'])

    def test_unchanged_best_is_rechecked_through_compiled_candidate_path(self):
        r = self.run_wrapper(str(self.flash))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][calls[0].index('--cand') + 1], calls[1][calls[1].index('--cand') + 1])

    def test_snowflake_pat_is_default_route_and_never_printed(self):
        r = self.run_wrapper(str(self.flash), OPENAI_API_KEY='fake-openai-key')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        call = self.claude_call()
        self.assertEqual((call['key'], call['url']), (PAT, SNOWFLAKE))
        self.assertEqual(call['argv'][call['argv'].index('--model') + 1], 'claude-opus-5[1m]')
        self.assertNoSecret(r, PAT, 'fake-openai-key')

    def test_explicit_endpoint_and_key_are_preserved(self):
        r = self.run_wrapper(str(self.flash), ANTHROPIC_BASE_URL='https://proxy.example/anthropic',
                             ANTHROPIC_API_KEY='fake-explicit-key')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        call = self.claude_call()
        self.assertEqual((call['key'], call['url']), ('fake-explicit-key', 'https://proxy.example/anthropic'))
        self.assertNotIn('--model', call['argv'])
        self.assertNoSecret(r, PAT, 'fake-explicit-key')

    def test_explicit_snowflake_endpoint_uses_explicit_key(self):
        r = self.run_wrapper(str(self.flash), ANTHROPIC_BASE_URL=SNOWFLAKE, ANTHROPIC_API_KEY='fake-explicit-key')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.claude_call()['key'], 'fake-explicit-key')

    def test_kda_key_is_reused_when_cortex_is_explicit(self):
        self.pat.unlink()
        r = self.run_wrapper(str(self.flash), ANTHROPIC_BASE_URL=SNOWFLAKE,
                             OPENAI_API_KEY='fake-kda-pat')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.claude_call()['key'], 'fake-kda-pat')
        self.assertNoSecret(r, 'fake-kda-pat')

    def test_key_without_endpoint_is_not_sent_to_snowflake(self):
        r = self.run_wrapper(str(self.flash), ANTHROPIC_API_KEY='fake-anthropic-key')
        self.assertStoppedBeforeGpu(r, 'ANTHROPIC_BASE_URL')
        self.assertNoSecret(r, PAT, 'fake-anthropic-key')

    def test_missing_pat_fails_before_gpu_work(self):
        self.pat.unlink()
        r = self.run_wrapper(str(self.flash), OPENAI_API_KEY='fake-openai-key')
        self.assertStoppedBeforeGpu(r, 'needs your PAT')
        self.assertNoSecret(r, 'fake-openai-key')

    def test_gpu_check_needs_no_credentials_and_does_not_call_claude(self):
        self.pat.unlink()
        r = self.run_wrapper('--check', str(self.flash))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertFalse((self.root / 'claude-call').exists())

    def test_smoke_reads_seed_log_and_rechecks_unchanged_kernel(self):
        r = self.run_wrapper('--smoke', str(self.flash))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('Smoke test passed (no optimization attempted)', r.stdout)
        call = self.claude_call()
        self.assertIn('SMOKE TEST ONLY', call['stdin'])
        self.assertIn('seed.log', call['stdin'])
        self.assertEqual(call['argv'][call['argv'].index('--allowedTools') + 1], 'Read,Write')
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][calls[0].index('--cand') + 1], calls[1][calls[1].index('--cand') + 1])

    def test_smoke_fails_if_claude_changes_best(self):
        r = self.run_wrapper('--smoke', str(self.flash), CHANGE_BEST='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Smoke test failed', r.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_missing_reference_stops_before_benchmark_or_agent(self):
        r = self.run_wrapper(str(self.flash), BAD_REFERENCE='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('missing reference fixture', r.stderr)
        self.assertFalse((self.root / 'calls').exists())

    def test_unsupported_case_stops_even_when_upstream_exits_zero(self):
        r = self.run_wrapper(str(self.flash), BAD_BENCH='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.root / 'claude-call').exists())

    def test_agent_reported_error_stops_before_final_benchmark(self):
        r = self.run_wrapper(str(self.flash), AGENT_ERROR='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(len((self.root / 'calls').read_text().splitlines()), 1)

    def test_missing_result_is_not_reported_as_success(self):
        r = self.run_wrapper(str(self.flash), NO_RESULT='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('did not write result.md', r.stderr)

    def batch(self):
        return next((self.root / 'runs').glob('experiment-*'))

    def test_paired_dataset_balanced_order_snapshots_and_measured_gain(self):
        import csv
        import hashlib
        r = self.run_wrapper(str(self.flash), KHA_TRIAL_DIR='', SEARCH_COUNT='2', CHANGE_BEST='1', REPEATS='2')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        batch = self.batch()
        manifest = json.loads((batch / 'experiment.json').read_text())
        self.assertEqual(manifest['gpu_index'], 0)
        self.assertEqual([v[1] for v in manifest['schedule'] if v[2] == 1].count('baseline'), 1)
        summary = list(csv.DictReader((batch / 'summary.csv').read_text().splitlines()))
        self.assertEqual(len(summary), 4)
        self.assertTrue(all(float(v['final_speedup_vs_seed']) == 1.25 for v in summary))
        self.assertTrue(all(v['qualifies'] == 'True' for v in summary))
        rows = [json.loads(line) for line in (batch / 'measurements.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 24)  # 4 sessions x (2 seed + 2 search + 2 final).
        self.assertEqual(sum(len(v['cases']) for v in rows), 72)
        self.assertEqual(len({v['kernel_sha256'] for v in rows if v['phase'] == 'seed'}), 1)
        for v in rows:
            trial = batch / v['trial_id']
            self.assertEqual(hashlib.sha256((trial/v['snapshot']).read_bytes()).hexdigest(), v['kernel_sha256'])
            if v['phase'] == 'search':
                self.assertIsNotNone(v['agent_elapsed_s'])
                self.assertTrue((trial / 'measurements' / v['measurement_id'] / 'notes.md').is_file())
        for trial in batch.glob('pair-*'):
            text = (trial / 'instructions.md').read_text()
            self.assertEqual('smallest test' in text, trial.name.endswith('guided'))
        call = self.claude_call()
        self.assertIn('--no-session-persistence', call['argv'])
        self.assertIn('--safe-mode', call['argv'])
        self.assertNoSecret(r, PAT)

    def test_search_budget_is_enforced_and_rejections_are_recorded(self):
        r = self.run_wrapper(str(self.flash), KHA_TRIAL_DIR='', SEARCH_COUNT='3', PAIRS='1')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for trial in self.batch().glob('pair-*'):
            ev = [json.loads(s) for s in (trial/'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['event'] == 'budget_exhausted' for e in ev), 1)
            self.assertEqual(sum(e['event'] == 'benchmark_start' and e['phase'] == 'search' for e in ev), 2)

    def test_failed_trials_stay_in_aggregate_dataset(self):
        r = self.run_wrapper(str(self.flash), KHA_TRIAL_DIR='', BAD_BENCH='1')
        self.assertNotEqual(r.returncode, 0)
        summaries = [json.loads(p.read_text()) for p in self.batch().glob('pair-*/summary.json')]
        self.assertEqual(len(summaries), 4)
        self.assertTrue(all(s['status'] == 'failed' and s['final_speedup_vs_seed'] is None for s in summaries))
        rows = [json.loads(s) for s in (self.batch()/'measurements.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(not r['valid'] for r in rows))

    def test_timeout_retains_unfinished_benchmark(self):
        r = self.run_wrapper(str(self.flash), KHA_TRIAL_DIR='', SEARCH_COUNT='1',
                             SLOW_SEARCH='1', SECONDS_PER_RUN='1', PAIRS='1')
        self.assertNotEqual(r.returncode, 0)
        for trial in self.batch().glob('pair-*'):
            summary = json.loads((trial/'summary.json').read_text())
            self.assertEqual(summary['agent_exit_code'], 124)
            ev = [json.loads(s) for s in (trial/'events.jsonl').read_text().splitlines()]
            interrupted = [e for e in ev if e.get('error', '').startswith('interrupted')]
            self.assertEqual(len(interrupted), 1)
            self.assertFalse(interrupted[0]['valid'])

    def test_comparison_rejects_resumed_sessions(self):
        r = self.run_wrapper(str(self.flash), '--resume', 'old-session', KHA_TRIAL_DIR='')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('only Claude --model and --effort', r.stderr)

    def test_missing_candidate_still_has_a_structured_failure_record(self):
        r = self.run_wrapper('--check', str(self.flash))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        trial = self.root / 'runs/single'
        r = subprocess.run([sys.executable, str(self.root/'experiment.py'), 'measure', 'missing.cu'],
                           cwd=trial, env=self.env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        record = json.loads((trial/'measurements/0002/measurement.json').read_text())
        self.assertFalse(record['valid'])
        self.assertIsNone(record['kernel_sha256'])
        self.assertIn('missing.cu', record['error'])
        self.assertTrue((trial/record['log']).is_file())


if __name__ == '__main__':
    unittest.main()
