# Frozen trap-world grid artifact (2026-09-17)

76 files, 190.3 MiB uncompressed: 73 grid records (results/grid/*.json) + 3 bootstrap records (results/bootstrap/*.json), generated 2026-09-16/17 (observed file mtimes: grid 2026-09-16 22:48 through 2026-09-17 00:37, bootstrap 00:50, all -0500 CDT) at pinned source 8ef2c3a467373052339d530ad950983b5afdae31.

## Where

- GitHub release asset (private repo, this repo): tag frozen-grid-20260917, file agentsem-trap-world-frozen-20260917.tar.gz
- Canonical local copy: ~/agentsem-archive/agentsem-trap-world-frozen-20260917.tar.gz (Mac)
- Archive SHA-256: e651f38623be83703cdc4ebde6d5f8dbc5b803e35384dd8bb996eaaf6fe3e572

## Integrity

GitHub immutable-releases is NOT enabled on this repo, so the release is a SHA-verified versioned private asset, not cryptographically immutable storage. Integrity guarantee is the pinned SHA-256 chain:
1. tar.gz SHA-256 above (GitHub asset digest verified identical at upload)
2. Per-file SHA-256 manifest of all 76 records, committed in-repo at results/FROZEN_GRID_SHA256.txt (sha256 per file) and reproducible from the unpacked tar via shasum -a 256 results/grid/*.json results/bootstrap/*.json

## Verification

```
gh release download frozen-grid-20260917 --repo abhaymettu/agentsem-trap-world
shasum -a 256 agentsem-trap-world-frozen-20260917.tar.gz   # expect e651f386...
tar -xzf agentsem-trap-world-frozen-20260917.tar.gz
```

Exact reproducibility verified 2026-09-28 on a clean checkout of pinned commit 8ef2c3a (read-only sampled repro, committed in-repo at scripts/reproduce_frozen_sample.py, script SHA-256 be41cae3624fe60b419592d891adcd0f5af69173fde1e08d4275919e8e3e4097):

- 8/8 stored hash checks match: analyzers_src_sha256, certificate_src_sha256, grid_hash, jev_policy_hash, params_hash, rules_hash, runner_src_sha256, surrogate_hash
- 12/12 sampled records re-run exactly equal (stored_record_hash == rerun_record_hash, first_diffs empty): cells A_k1.0_p0.5_g1.0 (cell_index 22), B_G2 (36), B_G3_r03 (38), B_G4 (40) x reps 0, 1, 19
- stored manifest: master_seed 20260916, manifest_sha256 8a0a2a22977ff9ef7f140a12e9e4844653e47beac94907a9ad4aee47f7255143; stored hash-file sha256 075c54cb0664d471db94971bf7a931113c288543083a865343d504c4d070ba3e
- environment: python 3.13.14, numpy 2.5.3, scipy 1.18.1, macOS arm64, pinned uv.lock 68eadfeba4d30611583cc6d0c70290941b4dbe689660e81598850c865c296e40
- A7 exploratory *_sample fields are stripped from stored records, so stored excerpts show shadow_raw_bias: null by design
- caveat from the repro receipt: repo_head_external_check_required = true (clean-checkout HEAD was verified against the pinned ref separately)

Roundtrip verified: release asset re-downloaded, unpacked, all 76 per-file SHA-256 matched, and the same 12/12 repro passed from the downloaded bytes.

## Retention

Keep indefinitely as the frozen evidence base for the trap-world benchmark. Not derived-regenerable cheaply (full grid run; sampled repro covers only the verification slice). Local archive + private release asset; treat as the canonical source for any future analysis or paper artifact appendix.
