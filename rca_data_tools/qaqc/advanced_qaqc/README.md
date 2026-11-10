# advanced_qaqc

Data products and QC flags computed by RCA that OOINet does not serve. Functions live in
`calculateFunctions.py`, are registered in `params/calculateCalls.csv` and assigned to streams
in `params/siteCalculations.csv`.

## Nitrate (NUTNR, shallow-profiler SUNAs)

Sites: `CE04OSPS-SF01B-4A-NUTNRA102`, `RS01SBPS-SF01A-4A-NUTNRA101`, `RS03AXPS-SF03A-4A-NUTNRA301`.

Runs at harvest, in this order: `nutnr_plant2023` → `nutnr_advanced_flags` →
`nutnr_qartod_recomputed`. The math is in `nitrate.py`.

### 1. Recomputed nitrate (`nutnr_plant2023`)

Outputs `dissolved_nitrate` [uM], `nutnr_fitting_function` (nitrate, baseline intercept,
baseline slope) and `nutnr_rmse`. NaN for dark frames and times without a calibration.

| Step | Reference |
|---|---|
| TCSS fit of nitrate plus linear baseline, 217-240 nm | Sakamoto 2009; Johnson & Coletti 2002 |
| Sea-salt extinction temperature correction | Plant 2023, Eq. 6 |
| Pressure correction, 2.6% per 1000 dbar | Sakamoto 2017 |

`recompute_nitrate(temp_model="sakamoto2009", pressure=False)` reproduces OOI L2
`salinity_corrected_nitrate`.

### 2. Per-sample tests (`nutnr_advanced_flags`)

Output `advanced_qc_flags`: one digit per test (1 pass, 3 flagged), ordered as
`attrs['tests_executed']`.

| Test | Flagged when | Reference |
|---|---|---|
| `CDOM` | A254 or A350 > 1.3 | CDOM note |
| `RMSE` | onboard fit RMSE > 0.001 | CDOM note; Argo 2.2.2.4 |
| `A240` | absorbance near 240 nm > 0.8 | Argo 2.2.2.3 |
| `saturation` | any fit-window pixel > 64,500 counts | Argo 2.2.2.2 |
| `stuck` | raw spectrum identical to a neighbour's | Argo test 13 |
| `lamp_temperature` | lamp > 35 °C | SUNA 8.3.2 |
| `lamp_hours` | lamp time > 750 h | SUNA 8.1.2 |
| `humidity` | internal humidity > 50% | SUNA troubleshooting |
| `dark` | dark value > 1000 counts | SUNA troubleshooting |
| `integration_time` | integration time factor > 1 | SUNA 8.4.1, 6.2.3 |
| `dark_noise` | temporal noise of the sample's dark frame > 12 counts | SUNA self-test; EMVA Eqs. 17-18 |
| `CTD_temperature`, `CTD_salinity`, `CTD_pressure` | CTD input fails QARTOD gross range (profiler CTD tables) | Argo 2.2.2.1 (RCA flags 3 for all) |
| `RMSE_recomputed` | recomputed fit RMSE > 0.001 | as `RMSE` |

Dark frames come from the separate `streamed-nutnr_a_dark_sample` stream.

### 3. QARTOD on recomputed nitrate (`nutnr_qartod_recomputed`)

Outputs `dissolved_nitrate_qartod_executed` (one digit per test) and
`dissolved_nitrate_qartod_results` (summary).

| Test | Flags | Limits | Reference |
|---|---|---|---|
| `climatology_test` | 3 outside the month x pressure-bin span | OOI qc-lookup | IOOS |
| `gross_range_test` | 4 outside fail span, 3 outside suspect span | OOI qc-lookup | IOOS |
| `flat_line_test` | 4 / 3 over the fail / suspect window | OOI qc-lookup | IOOS test 8 |
| `spike_test` | 3 where > 5 uM from the 5-point running median along the upcast | `nutnr_profile_tests.yaml` | Argo test 9 |
| `multi_variate_test` | 3 where a nitrate step between consecutive upcasts in a 5 dbar bin exceeds max(4 SD, 0.3 uM) and the temperature step is under 1 SD | `nutnr_profile_tests.yaml` | IOOS test 9 |
| `attenuated_signal_test` | per full-depth upcast: 4 where nitrate SD < 0.3 uM, 3 where below the site's suspect limit | `nutnr_profile_tests.yaml` | IOOS test 10 |

- OOI qc-lookup tables are the production ones built for `salinity_corrected_nitrate`.
- Upcasts (start to peak) come from the RCA
  [profileIndices](https://github.com/OOI-CabledArray/profileIndices).
- Profile-test statistics use samples that pass `CDOM`, `RMSE`, `A240`, `saturation`, `stuck`,
  `RMSE_recomputed` and gross range. The multi-variate test reads the previous `history_days`
  from the store.
- Flag values: 1 pass, 2 not evaluated, 3 suspect, 4 fail, 9 missing. Summary precedence follows
  ioos_qc `qartod_compare`: 4 > 3 > 1 > 2 > 9.
- `advanced_qc_flags` are not part of the summary.

### References

- **Johnson & Coletti 2002**: Deep-Sea Res. I 49:1291-1305.
- **Sakamoto 2009**: Sakamoto, Johnson and Coletti. Limnol. Oceanogr.: Methods 7:132-143.
- **Sakamoto 2017**: Sakamoto et al. Limnol. Oceanogr.: Methods 15:897-902.
- **Plant 2023**: Plant et al. Limnol. Oceanogr.: Methods 21:581-593.
- **IOOS**: U.S. IOOS (2018). Manual for Real-Time Quality Control of Dissolved Nutrients
  Observations, v1.1.
- **Argo**: Johnson et al. (2025). BGC-Argo quality control manual for nitrate concentration,
  v1.1. doi:10.13155/84370
- **CDOM note**: Sea-Bird Scientific (2017). CDOM Interference on Nitrate Measurements.
- **SUNA**: Sea-Bird Scientific (2025). SUNA V2 user manual, rev K.
- **EMVA**: EMVA (2021). EMVA 1288 Standard for Characterization of Image Sensors and Cameras,
  Release 4.0 General.
- **OOI qc-lookup**: https://github.com/oceanobservatories/qc-lookup/tree/master/qartod/nutnr
