# Module map

```
qaqc/
├── pipeline.py       CLI entry (qaqc_pipeline): picks sites/spans, launches flows
├── flow.py           Prefect flows and ECS deployment
├── plots.py          per-site plot task: read zarr window, calculate, decimate, render
├── dashboard.py      matplotlib plotting (scatter, profiles, ADCP, annotations)
├── decimate.py       LTTB and gap-aware coarsening
├── calculate.py      QartodRunner: runs QARTOD tests on plotted data
├── qartod.py         loads production and staged QARTOD tables
├── visual_data.py    camera image-count bar charts
├── index.py          dashboard index (qaqc_index)
├── notes.py          HITL annotation tables (generate_hitl_tables)
├── scheduler.py      triggers QAQC_dashboard GitHub workflows
├── advanced_qaqc/    derived parameters (calculateFunctions.py)
├── constants.py, utils.py
└── params/           instrument config, see the root README
```

Tests live in `tests/` at the repo root.
