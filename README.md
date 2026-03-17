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