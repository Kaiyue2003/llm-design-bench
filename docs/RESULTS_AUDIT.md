# Publication v1 Results Audit

This audit freezes the interpretation of the committed artifacts in
`reference_results/publication/`. It does not regenerate or overwrite those
artifacts. The checks in `tests/test_reference_results_audit.py` guard the
machine-verifiable claims below.

## Audit verdict

The publication v1 artifacts are internally consistent and suitable as a
historical three-method result set. They are **not** a reproduction of the
24-method comparison in the SPADE paper and must not be presented as an
unbiased all-task benchmark.

| Check | Result | Evidence |
| --- | --- | --- |
| Raw run cardinality | PASS | 240 rows = 3 methods x 10 tasks x 8 seeds |
| Unique run keys | PASS | one row per `(task, seed, optimizer)` |
| Seed coverage | PASS | paired seeds 38 through 45 for every task and method |
| Candidate budget | PASS | `K=128` in every raw row |
| Objective direction | PASS | LLM-DM uses negative cross entropy as maximization utility |
| Reference normalization | PASS | every raw score matches `(u - u_min) / (u_max - u_min)` |
| Aggregation | PASS | task means and sample SD (`ddof=1`) match the raw rows |
| Oracle accounting | PASS, limited | committed rows report zero optimization-time oracle queries; this audit cannot reconstruct process-level data access after the fact |
| Method scope | WARNING | only Best Logged, COM, and BDI are included |
| Task selection | WARNING | the nine synthetic tasks were selected using prior single-seed COM/BDI performance |
| Paper comparability | WARNING | publication v1 reports sample SD; the SPADE paper reports standard error |
| LLM-DM row count | OPEN | the adapter snapshot yields 454 usable observations, while the SPADE paper describes 472 designs |

## Frozen experiment contract

- Code commit: `8cc1f2466140648f7edad810e2ce8b5d75f5111a`
- Upstream data-recipes commit:
  `37269969a0957448d51622e0c083977bc5d260e8`
- Configuration fingerprint: `37c4cda71c6478d0`
- Methods: `best_logged`, `coms`, `bdi`
- Tasks: one LLM data-mixture task and nine selected synthetic functions
- Seeds: `38, 39, 40, 41, 42, 43, 44, 45`
- Recommendations per run: `128`
- LLM-DM visible split: global utility percentiles `[0, 40]`
- LLM-DM visible observations: `182`
- LLM-DM target fidelity: 1B parameters and 19,500 training steps
- Synthetic logged observations: 256 per task and seed
- Runtime: Python 3.11.15, PyTorch 2.12.0, deterministic algorithms enabled

The raw dataset therefore contains:

```text
1 LLM-DM task x 3 methods x 8 seeds       =  24 rows
9 synthetic tasks x 3 methods x 8 seeds  = 216 rows
total                                      = 240 rows
```

The seed manifest has 80 rows because it records one task/seed pairing, not
one method run.

The audited LLM-DM normalized-maximum results are:

| Entry | Mean | Sample SD | Interpretation |
| --- | ---: | ---: | --- |
| `D(best)` | 0.881 | 0.000 | best visible logged observation at recorded fidelity |
| Best Logged | 0.995 | 0.000 | selected logged mixture re-evaluated at target fidelity |
| COM | 0.992 | 0.007 | current PyTorch COM adaptation |
| BDI | 1.003 | 0.000 | current RBF-based BDI adaptation |

## What the table means

All tasks expose maximization utility. For LLM-DM, the original objective is
StackExchange cross-entropy loss and the adapter applies:

```text
utility = -cross_entropy_loss
```

The reference-normalized score is:

```text
(candidate utility - full logged minimum utility)
-------------------------------------------------
(full logged maximum utility - full logged minimum utility)
```

It is not clipped, so values above one are valid. The displayed uncertainty is
the sample standard deviation over eight runs. To compare a future result with
the SPADE paper's uncertainty convention, also report:

```text
standard error = sample standard deviation / sqrt(8)
```

`D(best)` and `Best Logged` are deliberately different for LLM-DM. `D(best)`
is the best optimizer-visible logged observation at its recorded fidelity.
Best Logged takes the selected mixture and re-evaluates it at the target
1B/19,500-step fidelity. They coincide on the synthetic tasks because those
tasks have no fidelity context.

## Method provenance

- **Best Logged** is a deterministic benchmark reference, even though the v1
  table repeats it on all eight seed rows.
- **COM** is the repository's native PyTorch reimplementation of the
  conservative-objective-model structure in `design-baselines`. It is not a
  byte-for-byte or numerical reproduction of the original implementation.
- **BDI** is a lightweight PyTorch adaptation using RBF kernel ridge
  regression. It replaces the original JAX/Neural Tangents infinite-width
  implementation and must be reported as **BDI adaptation**.

These v1 results predate the unified `OfflineBBOMethod` runner. They may be
loaded as legacy results, but new methods must be evaluated through the strict
oracle-separated runner before being added to a new comparison table.

## Selection bias

The synthetic subset is the union of the previously committed single-seed COM
and BDI winner in each function category. That makes the table a useful compact
descriptive result, but not an unbiased estimate of performance over the full
synthetic suite. A main benchmark must predeclare its task set independently of
the methods being compared. The v1 table should remain unchanged and be cited
as a historical/diagnostic table.

## The 454 versus 472 discrepancy

The SPADE paper describes LLM-DM as 472 five-dimensional designs. The current
adapter contract test expects 454 usable logged observations, and the v1 split
contains 182 observations (consistent with the adapter's 0--40 percentile
selection). The adapter skips a row when its history is empty or the selected
StackExchange metric is absent, so the most likely explanation is that 472 is
the upstream raw-design count while 454 is the metric-complete count. This is
an inference, not yet a verified row-level reconciliation.

Before claiming paper-level LLM-DM reproduction, add a data audit that records
for every upstream row whether it was retained and, if not, the exact exclusion
reason. Until then, reports must state both counts and call the discrepancy
unresolved.

## Integrity fingerprints

These SHA-256 values identify the audited v1 files:

| File | SHA-256 |
| --- | --- |
| `raw_runs.csv` | `e33544a5ef76f8397d9fc44e4be392f79cb27c43ef3070cc7fea515acfc140b8` |
| `task_summary.csv` | `a0a79887bd444e0c8d110d5bd01fef918f4dde4eba65b7e4397351170e3c34ab` |
| `rank_summary.csv` | `086e403bd234c1d2d5ed1c46647249d2d523af51e7f1c342ea727671f0e06682` |
| `seed_manifest.csv` | `7018a179609c580f1a0198ce12c2f8c4374fc52ffe418f12d639d44083a40946` |
| `run_metadata.json` | `5cecf228a67ba59d1444983ac33d3bf03beb7b0ee0e32abb15846bc395776eb7` |
| `d_best_summary.csv` | `8bf644237af47008d62dca05b9f754134bf32959ec147aee94f30b7dedf14157` |
| `seeded_benchmark_table.tex` | `1b18618b33d810d0495b8fb726cdf2199b1bc8c8f48b14ac7a00f9a1779bee21` |
| `README.md` | `333fb0aac2cc1d20d1c2fa69f4be7ffcfcd77892ab910d81ef717ee7acbd976f` |

## Rules for the next result set

1. Preserve `reference_results/publication/` as publication v1.
2. Use a new output directory and experiment identifier for the unified table.
3. Predeclare tasks, method configurations, seeds, and `K` before final runs.
4. Report mean, sample SD, and SE; label each column explicitly.
5. Keep `D(best)` outside stochastic method ranks unless the report explicitly
   defines another ranking policy.
6. Record failures and runtime instead of dropping failed seeds.
7. Label every port, wrapper, and adaptation according to its actual provenance.
