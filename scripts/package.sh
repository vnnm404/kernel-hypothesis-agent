#!/usr/bin/env bash
# Share source, pinned vendor files, instructions, and tests; exclude runs and local credentials.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
python3 - <<'PY'
from pathlib import Path
import tarfile
paths = ['kernel_agent', 'vendor', 'prompts', 'skills', 'experiments', 'scripts',
         'tests', 'docs', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
         'PROVENANCE.json', 'pyproject.toml', 'uv.lock', '.gitignore', 'run.sh']
def clean(info):
    parts = Path(info.name).parts
    if any(p in {'.git', '__pycache__', '.DS_Store', '.pytest_cache', '.env'} for p in parts):
        return None
    if info.name.endswith(('.pyc', '.ncu-rep', '.so')):
        return None
    return info
with tarfile.open('dist/kernel-hypothesis-agent.tar.gz', 'w:gz') as archive:
    for path in paths:
        if Path(path).exists():
            archive.add(path, arcname='kernel-hypothesis-agent/' + path, filter=clean)
print('dist/kernel-hypothesis-agent.tar.gz')
PY
