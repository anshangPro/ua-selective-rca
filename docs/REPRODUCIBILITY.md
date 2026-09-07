# Reproducibility protocol

1. Obtain RCAEval RE1/RE2 and the official CIRCA, RCD, BARO implementations under their licenses. Do not commit raw data.
2. Split by `fault_case` before fitting detector temperature or selective thresholds. Use train/validation/test case IDs rather than windows.
3. For each candidate boundary, run each external RCA tool on a window with identical pre/post durations and export the adapter JSON schema in the root README.
4. Fit `temperature` only on validation cases with known onset labels. Fix it before test evaluation.
5. Run all offsets in `configs/re1_template.json`, save a manifest with repository revisions, seed, device, CUDA version, GPU model, elapsed time, and peak GPU memory.
6. Report all methods and all fault cases. Include failed baseline runs and any runtime exclusions in the appendix.

## Required result tables

- Main accuracy: AC@1/3/5, MRR, Avg@5 by system and baseline.
- Robustness: mean, worst, variance, and area over boundary offsets.
- Selective RCA: coverage, selective risk, AURC, Top-k set recall, mean set size, abstention rate, and ECE.
- Ablation: single boundary, equal windows, posterior only, posterior plus stability, and selective output.

## GPU protocol

Use `cuda` where an external baseline or detector supports it. Record device name, total memory, CUDA/runtime version, and peak allocation. Supply a CPU smoke configuration for interface verification; do not treat it as a substitute for full experiments.
