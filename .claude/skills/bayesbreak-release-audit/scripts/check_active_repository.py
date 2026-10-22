#!/usr/bin/env python3
"""Check one active BayesBreak agent workflow and publication sources.

This checks source organization and removal of obsolete context, not numerical
correctness or scientific acceptance of existing figures.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
EXCLUDED_PATHS = [
    'docs/revision', 'docs/manuscripts/executive',
    'docs/manuscripts/presentation_handoffs',
    'docs/manuscripts/shared/handoffs',
    'results/phase6', 'results/phase7',
    'provenance',
    'scripts/tables/baselines_on_cached.py',
    'docs/manuscripts/shared/figures/results/baselines_metrics.json',
    'docs/manuscripts/shared/figures/results/baselines_metrics.txt',
    'docs/manuscripts/shared/metadata/experiment_protocols.json',
    'docs/manuscripts/book/SYNC_STATUS.md',
]
ACTIVE_FILES = [
    'CLAUDE.md', '.claude/README.md', '.claude/CURRENT_WORK.md',
    '.claude/RESULT_POLICY.md',
    '.claude/skills/bayesbreak-release-audit/SKILL.md',
    '.claude/skills/bayesbreak-release-audit/protocols/experiments.json',
    'docs/manuscripts/paper/main.tex', 'docs/manuscripts/book/main.tex',
]
BANNED_REFERENCES = re.compile(
    r'docs/revision/|results/phase[0-9]+/|EPR-BB-[0-9]+|REV-BB-[0-9]+'
    r'|CODE-BB-[0-9]+|shared/handoffs/|\\input\{.*phase[0-9]+'
)


def run():
    for name in EXCLUDED_PATHS:
        assert not (ROOT/name).exists(), f'obsolete path: {name}'
    for name in ACTIVE_FILES:
        assert (ROOT/name).is_file(), f'missing canonical file: {name}'
    assert (ROOT/'docs/manuscripts/paper/main.tex').read_text().find('11pt,onecolumn') >= 0
    assert list((ROOT/'.claude/skills').glob('*/SKILL.md')) and len(list((ROOT/'.claude/skills').glob('*/SKILL.md'))) == 1
    scan = [ROOT/'CLAUDE.md'] + sorted((ROOT/'.claude').rglob('*.md'))
    scan += sorted((ROOT/'docs/manuscripts/book').rglob('*.tex'))
    scan += sorted((ROOT/'docs/manuscripts/paper').rglob('*.tex'))
    scan += [ROOT/'README.md', ROOT/'docs/manuscripts/README.md']
    bad = []
    for path in scan:
        text = path.read_text(encoding='utf-8')
        for match in BANNED_REFERENCES.finditer(text):
            bad.append((str(path.relative_to(ROOT)), match.group(0)))
    if bad:
        raise AssertionError(f'obsolete contextual references: {bad[:12]}')
    print('PASS: one paper, one book, one Claude skill, no retired history or execution directories')

if __name__ == '__main__':
    try:run()
    except Exception as exc:
        print('FAIL:',exc,file=sys.stderr);raise SystemExit(1)
