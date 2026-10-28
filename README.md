# Regional Cabled Array Data Tools

Utilities for working with Regional Cabled Array Data. QAQC Dashboard back-end.

This repository contains code for utility modules and scripts for the RCA Data Team. The main author of this repo is [Wendi Ruef](https://github.com/wruef).

This is part of an effort to create a cohesive RCA Data Team Dashboard. Most of the current implementation were ported from the [QAQC_dashboard](https://github.com/OOI-CabledArray/QAQC_dashboard) repository.

# Development
Dependencies and the Python version are locked (`uv.lock`, `.python-version`):

```
uv sync                                   # local env
uv run --with pytest pytest -q            # offline test suite, incl. config integrity checks
uvx ruff@0.14.0 check rca_data_tools tests scripts
```

CI runs the tests and lint on every PR. For changes that touch plotting or data handling,
compare the plots against `main` (reads s3, so it is a dev tool, not CI):

```
python scripts/compare_renders.py <refDes> <span> [--params temperature] [--base main]
```

Prefect flows on ECS `git_clone` this repo's `main` at run time, so merged code is live on the
next run. The `rca-data-tools` image only supplies dependencies; rebuild it (stored in the RCA
AWS ECR) when they change. Authenticate with the AWS CLI, then push:

```
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
docker buildx build --platform linux/amd64,linux/arm64 -t public.ecr.aws/p0l4c7i2/rca-data-tools:latest --push .
```

# Previewing proposed QARTOD changes

Run the pipeline with staged (homebrew) QARTOD tables and publish the output to a
dashboard **archive** instead of the live dashboard. The `--prefix` flag routes
plots to `archives/internal/<slug>/QAQC_plots/`; without it, a `--homebrew-qartod`
sync is refused so live plots are never overwritten.

1. In the dashboard admin UI, create an internal archive (e.g. `proposed-qartod`).
   This registers the archive and snapshots the current dashboard as a baseline.
2. Generate staged-QARTOD plots into that archive prefix:
   ```
   qaqc_pipeline --stage1 --stage3 --cloud --s3-sync --homebrew-qartod --run \
     --span 30 --prefix archives/internal/proposed-qartod
   ```
3. Rebuild the archive's index:
   ```
   qaqc_index --cloud --prefix archives/internal/proposed-qartod
   ```
4. Log in to the dashboard → Internal tab → select the archive to review.

# Configuration Files 
`rca_data_tools/qaqc/params/`, the source of truth for what the pipeline plots and how.
`tests/test_config.py` checks the files reference each other consistently.

| File | Description |
|------|-------------|
| `sitesDictionary.csv` | Authoritative registry of all RCA instruments that are QAQCed by data team. Each row is one refDes with a `stage` column controlling pipeline routing and support status: `1` = standard sensors (CTD, fluorometer, pH, PCO2, ADCP, etc.), `2` = computationally intensive processing (VEL3D-C, OPTAA, SPKIR), `3` = cameras (CAMDS, CAMHD), `9` = acoustic and seismic instruments with alternate processing routines ie `ooi-hyd-tools` and `rca-echo-tools`. |
| `archiveDictionary.csv` | Same columns as `sitesDictionary.csv` (no `stage`) for retired instruments kept for archive plots. |
| `plotParameters.csv` | Maps instrument type to the ordered list of parameters that appear on its dashboard plot. |
| `variableMap.csv` | Maps canonical parameter names (e.g. `temperature`) to the actual NetCDF/Zarr variable names used across different instruments. |
| `variableParameters.csv` | Defines display properties for each parameter: expected min/max, profile min/max, axis label, color map, and whether it is static. |
| `multiParameters.csv` | Maps instruments whose parameters are multi-dimensional (e.g. spectral irradiance) to their individual sub-parameter names. |
| `deployedRanges.csv` | Restricts a coordinate dimension (e.g. ADCP bins) to a valid lower/upper index for the current deployment. |
| `maxCoordinateSizes.csv` | Maximum size of a coordinate per instrument (e.g. OPTAA wavelength, VADCP bins); data-vent pads to it so appends keep a fixed shape. |
| `localRanges.yaml` | Site-specific local parameter ranges for the `local` y-axis view. |
| `qartod_skip.yaml` | Lists parameters or instruments for which specific QARTOD tests (e.g. climatology, gross range) should be skipped. |
| `inheritance.yaml` | Declares, per stream, which input variables a derived parameter comes from (`downstream_from`). Not read by any code yet. |
| `compute_exceptions.yaml` | Lists streams that require non-default AWS compute resources (more vCPU/memory) than the standard Prefect workpool provides. |
| `harvest.yaml` | data-vent's daily harvest: streams harvested besides each stage 1/2 `zarrFile` (`extra_streams`), and non-default ECS sizes for harvest runs (`compute`). |
| `eventPresets.csv` | Instrument/parameter/overlay presets for the dashboard's event report page (e.g. `tsunami`). |
| `siteCalculations.csv` | Maps sites to the auxiliary calculations that should be run for them, and whether each site's calculations run during data-harvest or visualization pipeline (`runDuringHarvest`). |
| `calculateCalls.csv` | Defines each named calculation: the function key to call, its input parameters, optional kwargs, and return parameter name(s). |