# Additional-method simulation results

This snapshot contains simulations for all fourteen registered methods,
including the ten additional continuous PyTorch adaptations: CbAS, MINs,
DDOM, GABO, GTG, RGD, BONET, DEMO, ROOT and SPADE.

**These are two-epoch integration runs.** They establish execution and exact
replay under the recorded environments. Original-paper performance parity and
fully trained optimizer quality have not been established. The nine synthetic
tasks were selected from earlier COM/BDI category winners, so the comparisons
are descriptive and do not represent an unbiased benchmark-wide ranking.

## Results and scope

| Experiment | Coverage | Independent replay |
| --- | --- | --- |
| Native CPU simulations | 14 methods × 10 tasks × 8 seeds = **1,120 runs** | All 1,120 rows matched exactly |
| Docker CPU check | 14 methods × 2 tasks × 1 seed = **28 runs** | All 28 rows matched exactly |

The native tasks comprise the actual Data Recipes data-mixture simulator and
Ackley, Schaffer N. 2, Sum of Different Powers, Matyas, Power Sum, Rosenbrock,
Michalewicz, Hartmann 6-D and Shekel. Seeds are 38–45, with eight recommendations
per method/trial and 32 logged synthetic samples. The LLM-DM scores are
simulator predictions at the target 1B-parameter/19,500-step fidelity; these
experiments do not train new language models.

- [Scores: mean and sample standard deviation](SCORES.md)
- [Scores: mean and standard error](TABLE1_STYLE.md)
- [Observed seed ranges](SEED_RANGES.md)
- [All 1,120 raw per-seed results and resolved method settings](raw_runs.csv)
- [Aggregated task results](task_summary.csv) and [descriptive ranks](rank_summary.csv)
- [Seed manifest](seed_manifest.csv) and [run metadata](run_metadata.json)
- [Method source revisions and adaptations](METHOD_PROVENANCE.md)
- [Machine-readable verification and file hashes](snapshot_manifest.json)
- [Complete replay artifacts (ZIP)](reproduction_artifacts.zip)

The archive contains `native/first`, `native/replay`, `docker/first` and
`docker/replay`. Each includes its raw results, metadata, exact optimizer-visible
dataset arrays, generated candidates, and oracle evaluations. Docker batches
also include their container environment manifests. NPZ files contain numeric
arrays and are loaded with `allow_pickle=False`.

## Verify the uploaded data

From the repository root, in an environment with the project installed:

```sh
python -m zipfile -e reference_results/additional_methods_integration/reproduction_artifacts.zip results/uploaded_simulations
python scripts/verify_reproduction.py results/uploaded_simulations/native/first --compare results/uploaded_simulations/native/replay
python scripts/verify_reproduction.py results/uploaded_simulations/docker/first --compare results/uploaded_simulations/docker/replay
```

Both comparisons use `atol=0` and `rtol=0`. Verification checks complete
task/seed/method coverage, file hashes, array hashes and equality of candidates
and oracle evaluations. It compares the two native runs with each other and
the two Docker runs with each other; it does not claim cross-platform equality.

## Repeat the simulations

The implementation used is commit
`4cfe57c75b7cc5f2595167d45c601025b365c614`, with source-content SHA-256
`e5e31abcf027cbea2a983758de17b5886dfd0e4462761ec1256e9b919e2cba3c`.
The original metadata's dirty-working-tree flag is retained. The recorded
package source-content hash was checked against the committed implementation
before exporting this snapshot.

Use Data Recipes commit `37269969a0957448d51622e0c083977bc5d260e8` and the
package versions in [run metadata](run_metadata.json). With that checkout at
`/path/to/data-recipes`, run:

```sh
python -m llm_design_bench.publication_cli --all-methods --method-config reference_results/additional_methods_integration/method_settings.json --data-recipes-root /path/to/data-recipes --logged-samples 32 --recommendations 8 --epochs 2 --particle-steps 2 --bdi-steps 2 --method-steps 3 --torch-threads 1 --deterministic --no-resume --results-dir results/repeated_simulations
```

The default publication task list and seeds produce the ten-task/eight-seed
coverage above. Constructor overrides are frozen in
[method_settings.json](method_settings.json); each raw row records the fully
resolved configuration. Cross-machine or dependency changes may affect fresh
numerical results even with matching seeds.

For the independent two-task container check:

```sh
docker compose run --build --rm all-methods-smoke
```

The `all-methods` Compose service uses larger training budgets and therefore
does not reproduce this short-budget snapshot's scores. See the
[Docker guide](../../docs/DOCKER.md) and
[method implementation notes](../../docs/ADDITIONAL_METHODS.md).

The implementation passed 149 local tests (two optional-checkout tests skipped),
[Python 3.11/3.12 and package CI](https://github.com/Kaiyue2003/llm-design-bench/actions/runs/34440530875),
and the [Docker build and exact replay check](https://github.com/Kaiyue2003/llm-design-bench/actions/runs/34440530797).
