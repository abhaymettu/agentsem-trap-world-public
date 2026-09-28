"""Read-only sampled reproduction for agentSEM's frozen September 17 grid.

Run in a clean checkout pinned to commit 8ef2c3a, with project dependencies installed:
    python reproduce_frozen_sample.py --out /tmp/agentsem-repro-receipt.json
It reads results/, simulates selected cell/rep pairs, and writes only --out.
No Jev live calls. A7 exploratory *_sample fields are stripped from stored records.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import numpy as np
import scipy
from trapworld import grid, derive, run_grid

PINNED_REF = '8ef2c3a'
CELLS = ('A_k1.0_p0.5_g1.0', 'B_G2', 'B_G3_r03', 'B_G4')
SAMPLE_REPS = (0, 1, 19) # same rep IDs within distinct cells are NOT paired worlds


def clean_derived(record):
    r = json.loads(json.dumps(record))
    for block in r.get('analyzers', {}).values():
        for k in list(block):
            if k in derive.DERIVED_FIELDS or k.endswith('_sample'):
                del block[k]
    return r


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()


def compare(a, b, path='', atol=1e-8, rtol=1e-8):
    """Bounded first mismatches; separate numeric roundoff from structural drift."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f'{path} keys: stored-only={sorted(a.keys()-b.keys())}, rerun-only={sorted(b.keys()-a.keys())}']
        issues=[]
        for k in sorted(a):
            issues.extend(compare(a[k], b[k], path+'/'+k, atol, rtol))
            if len(issues)>=10: break
        return issues[:10]
    if isinstance(a, list) and isinstance(b, list):
        if len(a)!=len(b): return [f'{path}: lengths {len(a)} != {len(b)}']
        issues=[]
        for i,(x,y) in enumerate(zip(a,b)):
            issues.extend(compare(x,y,path+'/'+str(i),atol,rtol))
            if len(issues)>=10: break
        return issues[:10]
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if type(a)==type(b) and a==b else [f'{path}: {a!r} != {b!r}']
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return [] if math.isclose(a,b,rel_tol=rtol,abs_tol=atol) else [f'{path}: {a!r} != {b!r}']
    return [] if type(a)==type(b) and a==b else [f'{path}: {a!r} != {b!r}']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    args=p.parse_args()
    root=Path.cwd(); out=Path(args.out).resolve()
    if root in out.parents or out == root:
        raise SystemExit('--out must be outside the repository')
    stored_hashes=json.loads((root/'results/hashes.json').read_text())
    manifest=json.loads((root/'results/run_manifest.json').read_text())
    current_hashes=run_grid.compute_hashes()
    hash_checks={key:{'stored':stored_hashes.get(key),'current':v,'match':stored_hashes.get(key)==v}
                 for key,v in current_hashes.items()}
    if set(manifest['cells']) != set(manifest['reps_per_cell']):
        raise SystemExit('manifest cell/rep keys disagree')
    receipt={'pinned_ref_expected':PINNED_REF,'repo_head_external_check_required':True,
             'versions':{'python':platform.python_version(),'numpy':np.__version__,
                         'scipy':scipy.__version__},'manifest_versions':manifest['versions'],
             'manifest_master_seed':manifest['master_seed'],'manifest_grid_hash':manifest['grid_hash'],
             'computed_grid_hash':grid.GRID_HASH,'hash_checks':hash_checks,
             'stored_hash_file_sha256':hashlib.sha256((root/'results/hashes.json').read_bytes()).hexdigest(),
             'stored_manifest_sha256':hashlib.sha256((root/'results/run_manifest.json').read_bytes()).hexdigest(),
             'samples':[]}
    for cell_id in CELLS:
        if cell_id not in manifest['cells']: raise SystemExit(f'missing manifest cell {cell_id}')
        path=root/'results/grid'/f'{cell_id}.json'
        payload=json.loads(path.read_text())
        if payload['cell']['cell_id']!=cell_id: raise SystemExit(f'cell file mismatch {cell_id}')
        records=payload['records']; by_rep={r['rep']:r for r in records}
        if len(by_rep)!=len(records) or len(records)!=manifest['reps_per_cell'][cell_id]:
            raise SystemExit(f'record IDs/count disagree {cell_id}')
        for rep in SAMPLE_REPS:
            if rep not in by_rep: raise SystemExit(f'missing {cell_id}/{rep}')
            stored=clean_derived(by_rep[rep]); new=run_grid.run_replicate(
                grid.CELL_INDEX[cell_id],rep,bootstrap=0,jev_kind='fake')
            diffs=compare(stored,new)
            receipt['samples'].append({'cell_id':cell_id,'cell_index':grid.CELL_INDEX[cell_id],
              'rep':rep,'stored_record_hash':canonical_hash(stored),
              'rerun_record_hash':canonical_hash(new),'exact_equal':stored==new,
              'close_equal':not diffs,'first_diffs':diffs,
              'stored_excerpt':{'var_theta':stored.get('var_theta'),
                'naive_bias':(stored.get('analyzers',{}).get('naive_pooled') or {}).get('bias'),
                'anchor_bias':(stored.get('analyzers',{}).get('anchor_only') or {}).get('bias'),
                'shadow_raw_bias':(stored.get('analyzers',{}).get('agentsem') or {}).get('bias_B_raw'),
                'status_B':(stored.get('analyzers',{}).get('agentsem') or {}).get('status_B')}})
    receipt['pass']=(all(c['match'] for c in hash_checks.values()) and
       receipt['manifest_master_seed']==20260916 and
       receipt['manifest_grid_hash']==receipt['computed_grid_hash'] and
       all(s['close_equal'] for s in receipt['samples']))
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'pass':receipt['pass'],'samples':len(receipt['samples']),
        'hash_matches':sum(c['match'] for c in hash_checks.values()),
        'hash_total':len(hash_checks),'receipt':str(out)}))
    raise SystemExit(0 if receipt['pass'] else 1)

if __name__=='__main__': main()
