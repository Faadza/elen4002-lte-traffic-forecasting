# Prepared traffic data

The preparation pipeline provides the same hourly input history and observed target mask to every forecasting method. It builds 57 complete cell calendars, each containing 8,760 hours from 23 October 2017 at 00:00 to 22 October 2018 at 23:00 on the recorded clock. No timezone or physical traffic conversion is assumed.

`hourly.csv` contains 499,320 rows ordered by cell and timestamp:

| Column | Meaning |
|---|---|
| `cell` | Original cell identifier |
| `ts` | Recorded hourly timestamp |
| `traffic_observed` | Genuine traffic measurement; absent measurements remain NaN |
| `traffic` | Input after bounded causal filling; unresolved gaps remain NaN |
| `observed` | 1 for a genuine measurement and 0 otherwise |

The first three missing hours carry the last genuine input value. Leading gaps and the fourth and later missing hours remain unresolved. The counter continues across chronological splits. Filling never changes `traffic_observed` or `observed`, so filled values cannot become training or scoring targets.

For target hour `s`, the input interval is `[s-168, s)`. A target is eligible only when its measurement is genuine and all 168 preceding prepared inputs are finite. This excludes the target itself from its input window and gives every model the same target mask.

The intervals are half-open: development `[0, 6132)`, confirmation `[6132, 7446)` and final evaluation `[7446, 8760)`. Development folds score `[4116, 4788)`, `[4788, 5460)` and `[5460, 6132)`, using all preceding hours for fitting. LSTM stopping uses the last 336 hours of each fitting prefix.

| Period | Eligible targets | Genuine targets | Calendar opportunities |
|---|---:|---:|---:|
| Development after 168-hour history | 338,568 | 339,069 | 339,948 |
| Confirmation | 64,915 | 74,434 | 74,898 |
| Final evaluation | 74,474 | 74,474 | 74,898 |

These counts describe observation and input availability. They do not use held-out forecast errors or traffic-distribution diagnostics.

Main cell eligibility uses development observations, a finite positive population standard deviation, observed weekly pairs with a positive MASE scale and usable supervised windows. All 57 cells qualify. Nine tuning cells are selected independently from `[0, 4116)` using the nearest unused coefficient-of-variation quantiles from 0.1 to 0.9, with lexicographic ties. There is no arbitrary percentage-coverage cutoff.

`tools.check_prepared_data.load_validated_data(directory)` returns `(hourly, report)` and reconstructs an `eligible_target` column in memory. The loader uses round-trip CSV parsing to preserve the stored float64 values. Use `report["eligible_main_cells"]` and `report["tuning_cells"]` for the cell lists.

The checksum file distinguishes the current configuration from the original configuration recorded when the bundle was prepared. `runtime_config_sha256` pins the current `run_config.yaml`; `config_sha256` remains the original bundle configuration hash. The remaining preprocessing and bundle-file hashes are unchanged. The loader checks both configuration identities and reconstructs the calendar, observed mask, causal filling and target counts.

Additional bundle files contain missing runs, coverage for filling caps of 0, 1 and 3, development statistics, cell selection, source metadata and the validation report. The original bundle remains a fixed reference. New preparation outputs receive their own reports and must be reviewed separately.
