# rca-data-tools

QAQC plotting pipeline for the OOI Regional Cabled Array: zarr in `s3://ooi-data` →
QARTOD flags and calculations → matplotlib PNG/SVG → `s3://ooi-rca-qaqc-prod`, run as
Prefect flows on ECS (`scope-push-pool`).

## How it runs
- ECS flows `git_clone` this repo's `main` at run time, so merged code is live on the
  next run. The image (`rca-data-tools:latest`) only supplies dependencies; rebuild it
  only when they change.
- `rca_data_tools/qaqc/params/` (CSV/YAML) is the source of truth for instruments,
  parameters, ranges and compute sizes. `tests/test_config.py` checks they agree.
- Ownership: data-vent owns everything that talks to OOINet and writes data stores; this repo
  owns the instrument registry and everything that reads stores and renders.
- data-vent imports `HARVEST_STREAMS`, `HARVEST_COMPUTE`, `MAX_COORD_SIZES`, `PARAMS_DIR`, `S3_BUCKET`, `run_calculations_for_site`
  (from `calculate.py`), `load_site_calculations` and `get_s3_kwargs`; keep those names and signatures, and keep
  those modules free of plotting imports (`tests/test_data_vent_contract.py`).
- Pinned on purpose: `zarr<3` with `xarray<2026.4` (drop together in the zarr 3 migration),
  `pandas<3`, Python from `.python-version`.

## Conventions
- Prefer xarray; short comments; brevity over defensive error handling.
- Never `.load()` a full record (it OOMs at 16 GB). Loading a span window is fine.
- Heavy instruments get a bigger ECS deployment in `params/compute_exceptions.yaml`.

## Verifying changes
- `uv run --locked --with pytest pytest -q` and `uvx ruff@0.14.0 check rca_data_tools tests scripts`,
  as CI does on every PR (offline, seconds).
- Anything touching plotting or data handling: `python scripts/compare_renders.py <refDes> <span>`
  against `main`. Refactors must be byte-identical; intended changes should differ only in
  the files you expect. Pick instruments that exercise the path (fixed vs profiler vs ADCP,
  a deploy span, data with annotations/gaps).
- Pull ECS runtimes and logs with the Prefect client (`get_client().read_flow_runs`,
  `read_logs`) to compare runs before and after a change.
