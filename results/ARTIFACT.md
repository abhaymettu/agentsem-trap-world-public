# Frozen trap-world grid artifact (2026-09-17)

76 files, 190.3 MiB uncompressed: 73 grid records (results/grid/*.json) + 3 bootstrap records (results/bootstrap/*.json), generated 2026-09-17 (file mtimes Sep 17 04:18-04:24 CT) at pinned source 8ef2c3a467373052339d530ad950983b5afdae31.

## Where

- GitHub release asset (private repo, this repo): tag frozen-grid-20260917, file agentsem-trap-world-frozen-20260917.tar.gz
- Canonical local copy: ~/agentsem-archive/agentsem-trap-world-frozen-20260917.tar.gz (Mac)
- Archive SHA-256: e651f38623be83703cdc4ebde6d5f8dbc5b803e35384dd8bb996eaaf6fe3e572

## Integrity

GitHub immutable-releases is NOT enabled on this repo, so the release is a SHA-verified versioned private asset, not cryptographically immutable storage. Integrity guarantee is the pinned SHA-256 chain:
1. tar.gz SHA-256 above (GitHub asset digest verified identical at upload)
2. Per-file SHA-256 manifest of all 76 records (full list retained in the working archive /tmp/agentsem-inventory and reproducible from the unpacked tar via shasum -a 256 results/grid/*.json results/bootstrap/*.json)

## Verification

```
gh release download frozen-grid-20260917 --repo abhaymettu/agentsem-trap-world
shasum -a 256 agentsem-trap-world-frozen-20260917.tar.gz   # expect e651f386...
tar -xzf agentsem-trap-world-frozen-20260917.tar.gz
```
Exact reproducibility was verified 2026-09-28 on a clean checkout of the pinned commit: 8/8 file-hash matches and 12/12 sampled records (4 cells x reps 0,1,19) re-run exactly equal (agents_grid/proactive/rand_bias_json/trace_md5) with seed manager seed1234, python 3.12, pinned uv.lock 68eadfeb..., on darwin/arm64. Roundtrip verified: release asset re-downloaded, unpacked, all 76 per-file SHA-256 matched, and the same 12/12 repro passed from the downloaded bytes.

## Retention

Keep indefinitely as the frozen evidence base for the trap-world benchmark. Not derived-regenerable cheaply (20 reps x 12 cells). Local archive + private release asset; treat as the canonical source for any future analysis or paper artifact appendix.
