# Repository Guidelines

## Project Structure & Module Organization

This is a dependency-free Python research prototype for uncertainty-aware selective root-cause analysis. Package code lives in `src/ua_rca/`: keep domain models in `types.py`, pipeline orchestration in `pipeline.py`, and focused algorithms in modules such as `boundary.py`, `aggregation.py`, and `evaluation.py`. Adapter code for external RCA rankings belongs in `adapters.py`.

Keep executable entry points in `scripts/` (`run_synthetic_demo.py` and `run_case.py`), reusable configurations in `configs/`, and small, non-sensitive example inputs in `examples/`. `artifacts/` contains generated demo outputs; do not commit production telemetry. The `docs/REPRODUCIBILITY.md` protocol governs real experiments. Unit tests mirror package behavior under `tests/`.

## Build, Test, and Development Commands

Use Python 3.10 or later. Create or activate the repository virtual environment when one is available, then install the package in editable mode:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_synthetic_demo.py
python scripts/run_case.py --input examples/precomputed_case.json --output artifacts/case.json --config configs/re1_template.json
```

The test command runs the complete unit suite. The demo is a smoke test that produces an auditable JSON record in `artifacts/synthetic_demo/`.

## Coding Style & Naming Conventions

Use four-space indentation, standard-library-first imports, type hints for public APIs, and concise docstrings where behavior or units are non-obvious. Use `snake_case` for functions, variables, modules, and JSON fields; `PascalCase` for classes; and `UPPER_CASE` for constants. Keep algorithms deterministic where practical and surface inputs, factors, and decisions in the audit output instead of hiding them in script logic.

## Testing Guidelines

Write `unittest` tests named `test_*.py`; name methods `test_<expected_behavior>`. Add a focused regression test for every defect and cover both successful decisions and abstention/ambiguous cases. Run the full discovery command before opening a change. No coverage threshold is configured; prioritize boundary normalization, ranking aggregation, and decision behavior.

## Commit & Pull Request Guidelines

Git history is not available in this checkout, so use short imperative commit subjects, e.g. `Add boundary posterior regression test`. Keep commits narrowly scoped. Pull requests should explain the algorithm or interface change, list configuration/data assumptions, link the relevant issue when present, and include test output. Include representative JSON or screenshots only when output format or documentation changes.

## Security & Reproducibility

Never commit credentials, raw production telemetry, or licensed datasets. Split data by fault case before calibration, keep test thresholds fixed, and record seed, revisions, hardware, and runtime metadata for real experiments.
