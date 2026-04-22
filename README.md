# Regional Cabled Array Data Tools

Utilities for working with Regional Cabled Array Data. QAQC Dashboard back-end.

This repository contains code for utility modules and scripts for the RCA Data Team. The main author of this repo is [Wendi Ruef](https://github.com/wruef).

This is part of an effort to create a cohesive RCA Data Team Dashboard. Most of the current implementation were ported from the [QAQC_dashboard](https://github.com/OOI-CabledArray/QAQC_dashboard) repository.

# Development
The `rca-data-tools` imageis stored in the RCA AWS ECR. First authenticate with AWS CLI, then push changes. 

```
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
docker buildx build --platform linux/amd64,linux/arm64 -t public.ecr.aws/p0l4c7i2/rca-data-tools:latest --push .
```

# Configuration Files 
`rca_data_tools/qaqc/params/`

| File | Description |
|------|-------------|
| `sitesDictionary.csv` | Authoritative registry of all RCA instruments that are QAQCed by data team. Each row is one refDes with a `stage` column controlling pipeline routing and support status: `1` = standard sensors (CTD, fluorometer, pH, PCO2, ADCP, etc.), `2` = computationaly intensive processing (VEL3D-C, OPTAA, SPKIR), `3` = cameras (CAMDS, CAMHD), `9` = instruments with alternate processing routines. |
| `archiveDictionary.csv` | Same structure as `sitesDictionary.csv` for profiler/archive instruments. |
| `plotParameters.csv` | Maps instrument type to the ordered list of parameters that appear on its dashboard plot. |
| `variableMap.csv` | Maps canonical parameter names (e.g. `temperature`) to the actual NetCDF/Zarr variable names used across different instruments. |
| `variableParameters.csv` | Defines display properties for each parameter: expected min/max, profile min/max, axis label, color map, and whether it is static. |
| `multiParameters.csv` | Maps instruments whose parameters are multi-dimensional (e.g. spectral irradiance) to their individual sub-parameter names. |
| `deployedRanges.csv` | Restricts a coordinate dimension (e.g. ADCP bins) to a valid lower/upper index for the current deployment. |
| `localRanges.yaml` | Site-specific local parameters ranges |
| `qartod_skip.yaml` | Lists parameters or instruments for which specific QARTOD tests (e.g. climatology, gross range) should be skipped. |
| `inheritance.yaml` | Declares parameter dependencies so that a derived parameter |
| `compute_exceptions.yaml` | Lists streams that require non-default AWS compute resources (more vCPU/memory) than the standard Prefect workpool provides. |
| `siteCalculations.csv` | Maps sites to the auxiliary calculations that should be run for them, and whether each site's calculations run during data-harvest or visualization pipeline (`runDuringHarvest`). |
| `calculateCalls.csv` | Defines each named calculation: the function key to call, its input parameters, optional kwargs, and return parameter name(s). |