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


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='kernel wrapper ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('run.sh', 'prompt.md'):
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
  print('correct: 3/3   unsupported: 0   wrong: 0')
  print('perf over 3 configs: geomean=1.0x')
''')
        bins = self.root / 'bin'
        bins.mkdir()
        (bins / 'nvcc').write_text('#!/bin/sh\nexit 0\n')
        claude = bins / 'claude'
        claude.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
assert 'smallest test' in sys.stdin.read(), 'prompt was not passed on stdin'
Path(os.environ['CLAUDE_LOG']).write_text(json.dumps(sys.argv))
if not os.environ.get('NO_RESULT'):
 Path('result.md').write_text('Fixture result; no real GPU or model used.')
print(json.dumps({'type':'result', 'is_error': bool(os.environ.get('AGENT_ERROR'))}))
''')
        for p in bins.iterdir():
            p.chmod(0o755)
        self.env = os.environ | {'PATH': f'{bins}:/usr/bin:/bin', 'PYTHON_BIN': sys.executable,
                                  'CALL_LOG': str(self.root / 'calls'), 'CLAUDE_LOG': str(self.root / 'claude-call')}

    def run_wrapper(self, *args, **env):
        return subprocess.run(['bash', str(self.root / 'run.sh'), *args],
                              env=self.env | env, capture_output=True, text=True, timeout=20)

    def test_full_run_paths_with_spaces_and_existing_checkout_untouched(self):
        before = {str(p): p.read_bytes() for p in self.flash.rglob('*') if p.is_file()}
        r = self.run_wrapper(str(self.flash), '--model', 'configured-model')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        after = {str(p): p.read_bytes() for p in self.flash.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        calls = [json.loads(s) for s in (self.root / 'calls').read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][calls[0].index('--namespace') + 1], 'FA3_LIB_NAME')
        self.assertTrue(calls[1][calls[1].index('--cand') + 1].endswith('/best.cu'))
        self.assertIn('configured-model', json.loads((self.root / 'claude-call').read_text()))

    def test_gpu_check_does_not_call_claude(self):
        r = self.run_wrapper('--check', str(self.flash))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse((self.root / 'claude-call').exists())

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


if __name__ == '__main__':
    unittest.main()
