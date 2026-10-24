"""Independent read-only audit of sealed confirmation; never load final data or fit models."""

if __package__:
    from .audit_cli import run_audit, write_report
else:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from audit_cli import run_audit, write_report
    finally:
        sys.path.pop(0)


def _audit(paths):
    import hashlib
    import json
    import math
    from collections import Counter, defaultdict
    from datetime import datetime, timezone
    from pathlib import Path
    import numpy as np
    import pandas as pd

    BASE = paths.workspace
    ROOT = paths.repo
    DIRECTORY = paths.directory
    REPORT = paths.output
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    summary = read(DIRECTORY / 'summary.json')
    plan = read(DIRECTORY / 'plan.json')
    validation = read(DIRECTORY / 'validation-summary.json')
    selection = read(ROOT / 'docs/l1-l2-development-results/selection.json')
    protocol = read(ROOT / 'ensemble_study/protocol.json')
    assert summary['status'] == validation['status'] == 'passed'
    assert sha(DIRECTORY / 'plan.json') == summary['plan_sha256'] == validation['base_plan_sha256']
    assert sha(DIRECTORY / 'summary.json') == validation['base_summary_sha256']
    assert sha(ROOT / 'docs/l1-l2-development-results/selection.json') == summary['selection_sha256'] == plan['selection_sha256']
    assert summary['conditions'] == selection['conditions'] == plan['selection']['conditions']
    assert summary['primary_model'] == plan['primary_model'] == selection['primary_candidate'] == 'ensemble_l1_r0p1'
    assert summary['period'] == plan['period'] == 'confirmation'
    assert summary['uncertainty'] is None and not summary['final_settings_retuned']
    assert len(plan['code_sha256']) == 67
    for name, digest in plan['code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    cells = sorted(plan['cells'])
    assert len(cells) == 57 and len(set(cells)) == 57
    methods = [v['id'] for v in selection['conditions']]
    assert len(methods) == 11 and set(methods) == set(summary['methods'])
    metrics = ['MAE', 'MASE', 'RMSE']
    start, end = 6132, 7446
    comparisons_checked = 0
    max_difference = 0.0

    def mean(values):
        values = list(values)
        assert values
        return math.fsum(values) / len(values)

    def equal(a, b):
        nonlocal comparisons_checked, max_difference
        difference = abs(float(a) - float(b))
        comparisons_checked += 1
        max_difference = max(max_difference, difference)
        assert math.isclose(a, b, rel_tol=2e-13, abs_tol=1e-10), (a, b, difference)

    def match_dict(actual, expected):
        assert set(actual) == set(expected), (set(actual), set(expected))
        for key, value in actual.items():
            if isinstance(value, dict):
                match_dict(value, expected[key])
            elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                equal(value, expected[key])
            else:
                assert value == expected[key], (key, value, expected[key])

    def scores(actual, prediction, scale):
        errors = [float(p) - float(y) for p, y in zip(prediction, actual)]
        mae = mean(abs(e) for e in errors)
        return {'MAE': mae, 'MASE': mae / scale, 'RMSE': math.sqrt(mean(e * e for e in errors))}

    arrays, targets, scales, wholly, coverage, peak_targets, peak_counts = {}, {}, {}, {}, {}, {}, {}
    for cell in cells:
        entry = plan['series'][cell]
        path = DIRECTORY / entry['path']
        assert sha(path) == entry['sha256']
        with np.load(path, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        assert all(len(v) == end for v in data.values())  # Final values cannot be present.
        assert np.all(np.diff(data['timestamps']) == np.timedelta64(1, 'h'))
        arrays[cell] = data
        target = np.array([i for i in range(start, end)
                           if data['observed'][i] and np.isfinite(data['traffic'][i-168:i]).all()])
        assert target.tolist() == entry['targets']
        targets[cell] = target
        scales[cell] = mean(abs(float(data['traffic_observed'][i]) - float(data['traffic_observed'][i-168]))
                            for i in range(168, start) if data['observed'][i] and data['observed'][i-168])
        assert scales[cell] > 0
        wholly[cell] = np.array([data['observed'][i-168:i].all() for i in target])
        observed = int(data['observed'][start:end].sum())
        coverage[cell] = {'calendar_hours': end-start, 'eligible_targets': len(target), 'observed_targets': observed,
                          'fraction_calendar_scored': len(target)/(end-start), 'fraction_observed_scored': len(target)/observed}
        by_day = defaultdict(list)
        for i in range(start, end):
            by_day[str(data['timestamps'][i].astype('datetime64[D]'))].append(i)
        counts = dict(calendar_days_seen=len(by_day), complete_observed_days=0, qualifying_days=0,
                      tied_maximum_hours=0, excluded_incomplete_calendar_days=0,
                      excluded_missing_observation_days=0, excluded_ineligible_maximum_days=0)
        peaks = []
        lookup = {int(t): i for i, t in enumerate(target)}
        for positions in by_day.values():
            if len(positions) != 24:
                counts['excluded_incomplete_calendar_days'] += 1
                continue
            if not data['observed'][positions].all():
                counts['excluded_missing_observation_days'] += 1
                continue
            counts['complete_observed_days'] += 1
            maximum = max(data['traffic_observed'][positions])
            tied = [i for i in positions if data['traffic_observed'][i] == maximum]
            if not all(i in lookup for i in tied):
                counts['excluded_ineligible_maximum_days'] += 1
                continue
            counts['qualifying_days'] += 1
            counts['tied_maximum_hours'] += len(tied)
            peaks.append([lookup[i] for i in tied])
        assert peaks
        peak_targets[cell], peak_counts[cell] = peaks, counts

    jobs = {job['id']: job for job in plan['jobs']}
    assert len(jobs) == len(plan['jobs']) == len(summary['evidence']) == validation['successful_jobs'] == 741
    assert set(jobs) == set(summary['evidence'])
    raws, computed = {}, {}
    manifest_files = 0
    forecast_rows = 0
    attempt_counts = Counter()
    for job_id, job in jobs.items():
        assert job['fit_end'] == job['score_start'] == start and job['score_end'] == end
        assert job['predict']; assert job['repeat'] == (job['seed']-2025 if job['method']=='lstm' else 1)
        item = summary['evidence'][job_id]
        attempt = DIRECTORY / item['attempt']
        assert sha(attempt / 'complete.json') == item['completion_sha256']
        complete = read(attempt / 'complete.json')
        assert complete['status'] == 'passed' and complete['job'] == job and complete['plan_sha256'] == summary['plan_sha256']
        assert {str(p.relative_to(attempt)) for p in attempt.rglob('*') if p.is_file()} == set(complete['files_sha256']) | {'complete.json'}
        for name, digest in complete['files_sha256'].items():
            assert sha(attempt / name) == digest, (job_id, name)
            manifest_files += 1
        fit = read(attempt / 'fit.json')
        predict = read(attempt / 'predict.json')
        result = read(attempt / 'result.json')
        assert all(v['job'] == job and v['status'] == 'passed' for v in [fit, predict, result])
        assert fit['training']['fit_interval'] == [0, start]
        cell, method, seed = job['cell'], job['method'], job['seed']
        if method == 'lstm':
            assert seed in [2026, 2027, 2028]
            assert fit['training']['epochs'] == 23 and not fit['training']['early_stopping']
        else:
            assert seed is None
        frame = pd.read_csv(attempt / 'predictions.csv', float_precision='round_trip', keep_default_na=False, dtype={'seed': str})
        assert list(frame) == ['cell','method','seed','target_index','timestamp','actual','prediction_raw','prediction']
        assert frame.cell.eq(cell).all() and frame.method.eq(method).all() and frame.seed.eq('' if seed is None else str(seed)).all()
        target = targets[cell]
        data = arrays[cell]
        np.testing.assert_array_equal(frame.target_index.to_numpy(), target)
        np.testing.assert_array_equal(pd.to_datetime(frame.timestamp).to_numpy(), data['timestamps'][target])
        np.testing.assert_array_equal(frame.actual.to_numpy(), data['traffic_observed'][target])
        raw = frame.prediction_raw.to_numpy()
        assert np.isfinite(raw).all()
        prediction = np.maximum(0., raw)
        np.testing.assert_array_equal(frame.prediction.to_numpy(), prediction)
        key = method, cell, seed
        assert key not in raws
        raws[key] = raw
        forecast_rows += len(frame)
        score = scores(frame.actual, prediction, scales[cell])
        full = wholly[cell]
        full_score = scores(frame.actual.to_numpy()[full], prediction[full], scales[cell])
        clipping = {'prediction_count': len(raw), 'clipped_count': int(sum(raw < 0)), 'clipped_fraction': mean(raw < 0),
                    'mean_upward_adjustment': mean(prediction - raw)}
        errors = prediction - frame.actual.to_numpy()
        peak = {'status':'available', 'absolute_error':mean(mean(abs(float(errors[i])) for i in tied) for tied in peak_targets[cell]),
                'signed_error':mean(mean(float(errors[i]) for i in tied) for tied in peak_targets[cell]), **peak_counts[cell]}
        stored = summary['methods'][method]['cells'][cell]['seed_scores'][str(seed)]
        match_dict(score, stored['scores'])
        match_dict(full_score, stored['fully_observed_history_scores'])
        match_dict(clipping, stored['clipping'])
        match_dict(peak, stored['busiest_hour'])
        match_dict(score, predict['evaluation']['scores'])
        equal(scales[cell], predict['evaluation']['weekly_scale'])
        computed[key] = {'scores':score, 'fully_observed_history_scores':full_score, 'clipping':clipping, 'busiest_hour':peak}
        for existing in (DIRECTORY / 'jobs' / job_id).glob('attempt-*'):
            if (existing/'complete.json').exists(): attempt_counts['passed'] += 1
            else: attempt_counts['unsealed'] += 1
    assert attempt_counts == {'passed':741, 'unsealed':6}
    blend_equalities = 0
    for condition in selection['conditions']:
        if condition['kind'] != 'blend': continue
        method, objective, r = condition['id'], condition['objective'], condition['ridge_weight']
        for cell in cells:
            expected = r*raws['ridge',cell,None] + (1-r)*raws['lightgbm_'+objective,cell,None]
            np.testing.assert_array_equal(raws[method,cell,None], expected)
            blend_equalities += len(expected)

    percell, main_table = {}, []
    for method in methods:
        seeds = [2026,2027,2028] if method == 'lstm' else [None]
        stored = summary['methods'][method]
        assert stored['seed_count'] == len(seeds) and stored['cell_count'] == 57
        assert set(stored['cells']) == set(cells)
        for cell in cells:
            members = [computed[method,cell,seed] for seed in seeds]
            entry = stored['cells'][cell]
            sc = {m:mean(v['scores'][m] for v in members) for m in metrics}
            percell[method,cell] = sc
            match_dict(sc, entry['scores'])
            match_dict(coverage[cell], entry['coverage'])
            equal(scales[cell], entry['weekly_scale'])
            assert entry['fully_observed_history']['target_count'] == int(sum(wholly[cell]))
            match_dict({m:mean(v['fully_observed_history_scores'][m] for v in members) for m in metrics}, entry['fully_observed_history']['scores'])
            match_dict({k:mean(v['clipping'][k] for v in members) for k in members[0]['clipping']}, entry['clipping'])
            for m in ['absolute_error','signed_error']:
                equal(mean(v['busiest_hour'][m] for v in members), entry['busiest_hour'][m])
        macro = {m:mean(percell[method,c][m] for c in cells) for m in metrics}
        match_dict(macro, stored['scores'])
        for m in metrics:
            assert stored['contributors'][m] == cells and not stored['structural_exclusions'][m]
            equal(mean(mean(computed[method,c,seed]['fully_observed_history_scores'][m] for seed in seeds) for c in cells), stored['fully_observed_history'][m]['score'])
        for m in ['absolute_error','signed_error']:
            equal(mean(mean(computed[method,c,seed]['busiest_hour'][m] for seed in seeds) for c in cells), stored['busiest_hour'][m])
        for m in ['clipped_fraction','mean_upward_adjustment']:
            equal(mean(mean(computed[method,c,seed]['clipping'][m] for seed in seeds) for c in cells), stored['clipping'][m])
        main_table.append({'method':method, **macro, 'busiest_hour_absolute':stored['busiest_hour']['absolute_error'],
                           'busiest_hour_signed':stored['busiest_hour']['signed_error'], 'clipped_fraction':stored['clipping']['clipped_fraction'],
                           'fully_observed_history_MASE':stored['fully_observed_history']['MASE']['score']})
    standalones = protocol['standalone_order']
    comparator = min(standalones, key=lambda m: (summary['methods'][m]['scores']['MASE'],standalones.index(m)))
    assert comparator == summary['primary_comparator'] == summary['comparator_selection']['method'] == 'lightgbm_l1'
    assert summary['comparator_selection']['exact_tied_candidates'] == [comparator]
    primary = summary['primary_model']
    for method in methods:
        for metric in metrics:
            saved = summary['primary_minus_comparators'][method][metric]
            differences = {c:percell[primary,c][metric]-percell[method,c][metric] for c in cells}
            for c in cells: equal(differences[c],saved['cell_differences'][c])
            equal(mean(differences.values()),saved['primary_minus_comparator'])
            equal(summary['methods'][primary]['scores'][metric]-summary['methods'][method]['scores'][metric],saved['absolute_difference'])
            equal(100*(summary['methods'][primary]['scores'][metric]/summary['methods'][method]['scores'][metric]-1), saved['percentage_difference_vs_comparator'])
            assert sum(v<0 for v in differences.values()) == saved['improved_cell_count']
            equal(sum(v<0 for v in differences.values())/57,saved['fraction_cells_improved'])
    assert summary['designated_primary_comparison']['metrics'] == summary['primary_minus_comparators'][comparator]
    comparative=[]
    for candidate, reference in [(primary,comparator),('ensemble_l2_r0p2','lightgbm_l2'),('ensemble_l1_r0p5','lightgbm_l1'),('ensemble_l2_r0p5','lightgbm_l2'),('ensemble_l1_r0p5',primary),(primary,'ensemble_l2_r0p2')]:
        comparative.append({'candidate':candidate,'reference':reference,
            'percentage_difference_negative_is_better':{m:100*(summary['methods'][candidate]['scores'][m]/summary['methods'][reference]['scores'][m]-1) for m in metrics},
            'improved_cells':{m:sum(percell[candidate,c][m]<percell[reference,c][m] for c in cells) for m in metrics}})
    report = {
        'status':'passed', 'scope':'Independent confirmation arithmetic, chronology, common cohorts, diagnostics, raw blend identity and frozen comparator selection',
        'created_at_utc':datetime.now(timezone.utc).isoformat(), 'script_sha256':sha(Path(__file__)),
        'directory':str(DIRECTORY), 'plan_sha256':sha(DIRECTORY/'plan.json'), 'summary_sha256':sha(DIRECTORY/'summary.json'),
        'validation_summary_sha256':sha(DIRECTORY/'validation-summary.json'), 'selection_sha256':summary['selection_sha256'],
        'independence':'Uses stdlib math.fsum plus NumPy/pandas I/O; does not import any production scoring, selection, fitting or audit module.',
        'checks':{'unchanged_original_runtime_files':67,'cells':57,'conditions':11,'accepted_jobs_and_csvs':741,'unsealed_attempts_preserved':6,
                  'manifest_files_hashed':manifest_files,'prediction_rows':forecast_rows,'exact_raw_blend_equalities':blend_equalities,
                  'numeric_equalities':comparisons_checked,'maximum_absolute_numeric_difference':max_difference,
                  'absolute_tolerance':1e-10,'relative_tolerance':2e-13,'lstm_seed_results':171,'lstm_epochs':23,
                  'common_masks_prefix_scales_seed_score_means_clipping_peaks_and_history_sensitivity':True},
        'coverage':{'eligible_cell_hours':sum(len(targets[c]) for c in cells),'observed_cell_hours':sum(coverage[c]['observed_targets'] for c in cells),
                    'calendar_cell_hours':57*(end-start),'fully_observed_history_targets':sum(int(sum(wholly[c])) for c in cells),
                    'minimum_cell_targets':min(map(len,targets.values())),'maximum_cell_targets':max(map(len,targets.values()))},
        'frozen_primary':primary,'selected_standalone_comparator':comparator,'comparator_exact_tie':False,
        'main_and_supporting_table':main_table,'comparisons':comparative,
        'fitting_performed':False,'final_values_or_errors_read':False,
        'limitations':['This is confirmation, not the final evaluation; the final primary uncertainty intervals and formal complete-system cost remain pending.',
            'L1 50/50 has lower confirmation MASE than the frozen primary; this supporting result does not permit changing the development-frozen primary or weight.',
            'MASE, MAE and macro-RMSE rank objectives differently. Macro-RMSE is the equal-cell mean of cell RMSE, not pooled RMSE.',
            'All conditions share the eligible target mask; missing observations/history exclude some confirmation calendar hours. The observed-history sensitivity is reported separately.',
            'MASE above one compares current errors with the historical fitting-prefix weekly scale; it does not by itself imply losing to the current-period weekly-naive forecasts.',
            'This audit verifies saved arrays and arithmetic without reloading native models or recreating hardware readiness and timing. The production validation summary and separate operational audit cover those checks.',
            'Six unsealed attempts are preserved. The amended standardized LSTM batch-consistency validation remains required in the final protocol; this audit does not change that rule.',
            'Historical adaptive experimentation, unequal cumulative tuning budgets and earlier descriptive exposure remain disclosed.']}
    write_report(REPORT, report, sort_keys=False)
    print(json.dumps({'report':str(REPORT),'sha256':sha(REPORT),'checks':report['checks'],'coverage':report['coverage'],'comparisons':comparative,'table':main_table},indent=2))

def main(argv=None):
    run_audit(__file__, "confirmation", _audit, argv)


if __name__ == "__main__":
    main()
