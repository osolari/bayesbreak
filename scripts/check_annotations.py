"""Validate one-to-one bibliographic annotations without historical manifests."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[1]


def check_annotations() -> dict:
    base=ROOT/'docs/manuscripts/shared/bibliography'
    bibliography=(base/'references.bib').read_text(encoding='utf-8')
    keys=set(re.findall(r'@[A-Za-z]+\s*\{\s*([^,\s]+)',bibliography))
    paths=list((base/'annotated_entries').glob('*.tex'))
    annotations={p.stem for p in paths}
    empty=[p.name for p in paths if 'bayesbreak' not in p.read_text(encoding='utf-8').lower()]
    missing=sorted(keys-annotations)
    extras=sorted(annotations-keys)
    return {
        'passed':len(keys)==38 and len(paths)==38 and not (missing or extras or empty),
        'counts':{'bibliography_keys':len(keys),'annotation_files':len(paths)},
        'missing_annotations':missing,'orphaned_annotations':extras,
        'annotations_without_project_relationship':empty,
    }


if __name__=='__main__':
    result=check_annotations();print(json.dumps(result,indent=2,sort_keys=True));
    raise SystemExit(0 if result['passed'] else 1)
