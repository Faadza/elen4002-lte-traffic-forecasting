# Running and checking the code

Use the interpreter and packages in [environment setup](environment-setup.md).
The [data instructions](../data/README.md) describe the authorised input CSV and
the original prepared bundle. Data, saved models and experiment results are local
inputs; they are not distributed in this repository.

## Short checks

From the repository root:

```bash
.venv/bin/python tools/run_cpu.py -m pytest tests -q -rs
```

These tests use invented data and small model fits. They check calculations,
selection boundaries, storage and execution guards. They do not reproduce the
reported real-data accuracy or formal timings. One real-backend CPU test is
Linux-only and is skipped on macOS. The experiment runners use Unix facilities;
the Windows package candidate is not a claim of complete Windows runner support.

## Development and historical records

`tools.run_development_experiments` prepares and executes new development plans
against the current configuration and source. Its resume checks remain strict:
a different code inventory, interpreter, machine or package environment is not
the same run. Use a new output directory for a new experiment.

The Ridge extension and L1 investigation also read the completed original
development experiment. That archive has its own fixed configuration, source,
plan and result identities. The historical verifier checks those original
identities separately from the exact reviewed changes listed in
[source compatibility](source-compatibility.json). It does not replace hashes in
archived files or admit arbitrary edits. A read-only archive check is not
permission to continue training on a different host.

The main entrypoints are:

| Purpose | Python module |
|---|---|
| Original development search | `tools.run_development_experiments` |
| Ridge regularisation extension | `adaptation.run_ridge_extension` |
| L1 objective investigation | `l1_revision.run_experiment` |
| Independent blend replay | `blend_revision.replay` |
| Held-out evaluation | `final_evaluation.run` |
| L1/L2 ensemble development | `ensemble_study.development` |
| Ensemble confirmation and final evaluation | `ensemble_study.evaluation_run` |
| Published LSTM validation continuation | `validation_revision.continuation` |
| Published final-stage continuation | `final_stage.run` |

For arguments, use the selected module's `--help` through `tools/run_cpu.py`.
Preparation, fitting, recovery and study launchers are not installation checks.
Fitting additionally requires the relevant explicit execution flag and native
host-readiness conditions.

## Sealed final and recovery evidence

The final and recovery controllers preserve the identities of a particular
completed study. Use an authentic archived runtime whose **complete source-file
set and bytes** match its frozen manifests, with the original records and
interpreter. A copy of the current public code is not that archived runtime.
An extra Python file in an inventoried package can invalidate its identity even
when all the original files are unchanged.

The required private layout includes:

```text
archive-workspace/
  ensemble-study-review/
    <runtime source and configuration matching the frozen manifests>
    docs/commands/
      confirmation-recovery-v1.json
      confirmation-validation-v2.json
      final-evaluation-v1.json
      run_ensemble_confirmation_mac.sh
      resume_ensemble_confirmation_mac.sh
      resume_ensemble_confirmation_v2_mac.sh
      run_ensemble_final_mac.sh
    docs/l1-l2-confirmation-results/
      arithmetic-audit.json
      integrity-audit.json
    results/local_runs/l1-l2-independent-ensemble/
      development-selection/
      confirmation/
      final/
  context/reconciliation-review/
    l1-l2-evaluation-publication-2026-10-09.json
    confirmation-recovery-publication-2026-10-09.json
    confirmation-validation-publication-2026-10-09.json
    final-evaluation-publication-2026-10-09.json
    lstm-float32-numerical-audit-2026-10-09/
```

Retain every file named in the authentic inventories, including plans, checksums,
selections, series, attempt/model/prediction files, completion records, sessions
and diagnostic records. This layout shows the main locations; it is not a
replacement manifest. Original-development and L1 result directories referenced
by the plans must also be available.

Some records contain absolute paths. Moving a folder does not update those
identities. The publication records verify the archived runtime and cannot be
reused as proof that a changed runtime was published. Do not regenerate a
`passed` receipt, change old operator labels, or replace hashes to make a check
accept different evidence.

The v2 continuation includes the published LSTM numerical rule. The older v1
recovery source is retained for its historical role and only classifies the
original CPU-readiness failures. It is not a substitute for v2.

Formal continuation also requires the recorded host/interpreter/packages and
the declared power/background conditions. Read-only evidence inspection and a
new experiment on another machine are separate operations; their timing results
must not be treated as one unchanged run.

## Independent audit commands

The five dated scripts in `docs/verification/` accept:

- `--repo`: runtime source checkout; defaults to this checkout.
- `--workspace`: historical diagnostic workspace; defaults to the chosen
  checkout's parent.
- `--directory`: saved confirmation or final run; defaults to that period under
  the chosen checkout's `results/local_runs/l1-l2-independent-ensemble/`.
- `--output`: required fresh JSON report, or a fresh directory for the saved
  LSTM-kernel audit.

For the sealed historical study, choose the matching archived runtime with
`--repo`. The scripts retain the original study-specific hashes, counts,
conditions and numerical assertions; they are not generic audits of arbitrary
new experiments. `--workspace` selects the diagnostic lookup location. It does
not relocate publication receipts owned by imported runtime modules or rewrite
paths embedded in saved plans.

Example, after the authentic runtime and evidence have been checked:

```bash
ELEN_ORIGINAL_PYTHON=/path/to/the/original/configured/.venv/bin/python
"$ELEN_ORIGINAL_PYTHON" tools/run_cpu.py docs/verification/audit_final_integrity_2026_10_10.py \
  --repo /path/to/archive-workspace/ensemble-study-review \
  --workspace /path/to/archive-workspace \
  --directory /path/to/archive-workspace/ensemble-study-review/results/local_runs/l1-l2-independent-ensemble/final \
  --output /path/to/new-audits/final-integrity.json
```

Set `ELEN_ORIGINAL_PYTHON` to the exact interpreter recorded in the archived
configuration; a new virtual environment with the same packages has a different
identity. Use a separate new output location; an existing report is never
replaced. The arithmetic scripts do not start work merely when imported, and
the five scripts reject optimised Python (`-O`) because it disables assertions.

The separate `final-independent-cost-2026-10-10.py` takes `--repo` and
`--output`; it also requires the saved study evidence. Arithmetic/bootstrap
audits and the saved-kernel audit can take substantial time. The latter performs
saved-model inference. These full audits are not part of the short unit-test
command above.
