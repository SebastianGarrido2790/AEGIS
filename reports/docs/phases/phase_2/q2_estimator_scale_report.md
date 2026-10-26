# Q2 Estimator Scale Qualification Report

**Executed:** 2026-10-26  
**Seed:** 42 (single pilot seed; multi-seed evaluation is Stage 9/10 scope)  
**Decision:** n* = 413,169, nuisance = `gbt`,
cv = 5

## Method

Each requested N is a stratified total sample followed by an 80/20 stratified train/test
split. CausalForestDML was fit only on the training partition with 5-fold cross-fitting.
It used 100 trees (library default) and all available CPUs. GBT nuisance models used
scikit-learn defaults; linear nuisance models used LogisticRegressionCV (max_iter=1,000)
and ElasticNetCV (max_iter=10,000), each with 5-fold regularization CV and default
regularization grids. The run used 12 logical CPUs.
The test partition was scored against the isolated truth sidecar; truth values were not
passed to estimator fitting. Predicted and true segment effects use the production
ratio-of-sums aggregation (`sum(tau) / sum(q0)`). Baseline renewal probability was constant
within each of the 35 configured segments in this truth sidecar, so the ratio-of-sums segment
mean and the `ate_inference` segment-average estimand coincide. Segment confidence intervals
are obtained from `ate_inference` on each held-out segment. No grid cells were skipped.

Peak process RSS was sampled from the operating system while each cell ran. Peak
`tracemalloc` memory is also recorded; unlike RSS, it does not include all native-library
allocations and is not used for the RAM budget decision.

## Results

| N | Nuisance | Train/test | Wall time (s) | Peak RSS (MB) | Peak traced memory (MB) | Segment Spearman | 95% CI coverage | Segments |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 25,000 | gbt | 20,000/5,000 | 24.4 | 601.3 | 88.5 | -0.1217 | 1.000 | 35 |
| 25,000 | linear | 20,000/5,000 | 51.5 | 624.6 | 90.5 | 0.0176 | 1.000 | 35 |
| 100,000 | gbt | 80,000/20,000 | 113.4 | 973.3 | 352.1 | 0.5139 | 1.000 | 35 |
| 100,000 | linear | 80,000/20,000 | 94.1 | 1057.9 | 377.2 | 0.5960 | 1.000 | 35 |
| 250,000 | gbt | 200,000/50,000 | 363.8 | 1735.5 | 906.5 | 0.8753 | 1.000 | 35 |
| 250,000 | linear | 200,000/50,000 | 120.6 | 1901.9 | 913.2 | 0.8520 | 1.000 | 35 |
| 413,169 | gbt | 330,535/82,634 | 514.8 | 2534.3 | 1455.6 | 0.9014 | 1.000 | 35 |
| 413,169 | linear | 330,535/82,634 | 171.4 | 2792.3 | 1434.9 | 0.9031 | 1.000 | 35 |

## Selection

The full-data runtime budget is 30.0 minutes and the RSS
ceiling is 39262.1 MB (80.0% of 49077.6 MB
installed RAM). 2 full-data fit(s) met both limits, so the full 413,169-policy scale was adopted. At N=413,169, linear scored 0.9031 and GBT scored 0.9014 (gap 0.0017); because the gap is within 0.02, the approved tie-break selects GBT. The selected fit ran in
8.58 minutes and peaked at 2534.3 MB
(5.2% of installed RAM).

## Frozen configuration

```yaml
tier1_ml:
  estimator:
    training_sample_size: 413169
    nuisance_model: "gbt"
    cv_folds: 5
    runtime_budget_minutes: 30.0
```

## Reproducibility

- Training table SHA-256: `f08918631e891203f845f7107d9e7c33572d817b16f81d75b7f737424471cdb8`
- Truth sidecar SHA-256: `21b1713ba4089c536aea6d0b47770fb8683cd10f26d5efd7d7e64a7e231cda1e`
- Segment definitions SHA-256: `079407763318d30e8cf99a5568602ea6a24cf160d21b143d4d778455917edf8f`
