"""Numerical amendment checks and preserved-attempt continuation safeguards."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

from validation_revision import measurement as m
from validation_revision import continuation as c
from tools.development_store import write_json_atomic, begin_attempt, complete_attempt


class MeasurementTests(unittest.TestCase):
    def test_fixed_precision_rule_and_translation_units(self):
        self.assertEqual(m.TOLERANCE, 0.0003452669770922512)
        for mean, sd in [(0., 1.), (400., 450.), (-200., .25)]:
            audit = m.check_origins(mean+sd*np.array([-.9, 0., 2.]),
                mean+sd*np.array([-.9+1e-5, 1e-5, 2.+1e-5]), [-.9, 0., 2.], mean, sd)
            m.verify_audit(audit, 3)
            self.assertLess(audit['differences']['single_vs_batch']['maximum_bound_fraction'], .1)

    def test_wrong_raw_even_if_both_clip_to_zero_is_rejected(self):
        with self.assertRaises(AssertionError):
            m.check_origins([-3.], [-3.1], [-3.], 0., 1.)

    def test_both_kernels_wrong_together_are_rejected(self):
        with self.assertRaises(AssertionError):
            m.check_origins([1.], [1.], [0.], 0., 1.)

    def test_nonfinite_shape_and_scaler_rejected(self):
        for args in [([np.nan], [1.], [1.], 0., 1.), ([1.], [1., 2.], [1.], 0., 1.),
                     ([1.], [1.], [1.], 0., 0.), ([1.], [1.], [1.], np.inf, 1.)]:
            with self.assertRaises(ValueError):
                m.check_origins(*args)

    def test_workloads_and_timers_preserved_reference_outside_timers(self):
        calls, timer, audited = [], [False], []
        def predict(indices):
            calls.append(indices.tolist())
            raw = np.array(indices, dtype=float)-1.
            return raw, np.maximum(0., raw)
        def elapsed(fn):
            timer[0] = True
            try:
                return fn(), .01
            finally:
                timer[0] = False
        def reference(origins):
            self.assertFalse(timer[0])
            audited.append(True)
            return origins-1.
        with patch.object(m, 'elapsed_call', side_effect=elapsed):
            result = m.measure_lstm(predict, [0, 1, 2], [0, 2], mean=0., sd=1., reference=reference)
        self.assertEqual(calls, [[0, 1, 2]]*4 + [[0], [2]]*4)
        self.assertEqual(audited, [True])
        np.testing.assert_array_equal(result['raw'], [-1., 0., 1.])
        self.assertEqual(result['timings']['single_origin_seconds'], [[.01, .01]]*3)
        m.verify_audit(result['timings']['numerical_validation'], 2)

    def test_repeat_changes_rejected_exactly(self):
        number = [0]
        def predict(indices):
            number[0] += 1
            raw = np.ones(len(indices)) + number[0]*1e-12
            return raw, raw
        with self.assertRaisesRegex(AssertionError, 'Repeated identical'):
            m.measure_lstm(predict, [1, 2], [1], mean=0., sd=1., reference=lambda x: x)

    def test_wrong_clipping_rejected(self):
        with self.assertRaises(ValueError):
            m.measure_lstm(lambda x: (-np.ones(len(x)), np.ones(len(x))), [1, 2], [1],
                           mean=0., sd=1., reference=lambda x: x)

    def test_empty_common_origins_remain_unavailable(self):
        result = m.measure_lstm(lambda x: (np.ones(len(x)), np.ones(len(x))), [1, 2], [],
            mean=0., sd=1., reference=lambda x: self.fail('No independent origins exist'))
        self.assertFalse(result['timings']['single_vs_batch_checked'])
        self.assertEqual(result['timings']['single_origin_status'], 'unavailable_no_origins_common_to_all_cells')

    def test_forged_audit_rule_or_statistics_rejected(self):
        audit = m.check_origins([1.], [1.], [1.], 0., 1.)
        for key, value in [('rule', 'other'), ('rtol', 1.), ('origin_count', 2), ('differences', {})]:
            changed = copy.deepcopy(audit); changed[key] = value
            with self.assertRaises(ValueError):
                m.verify_audit(changed, 1)

    def test_claimed_origin_count_cannot_hide_short_arrays(self):
        audit = m.check_origins([1.], [1.], [1.], 0., 1.)
        audit['origin_count'] = 100
        with self.assertRaisesRegex(ValueError, 'arrays do not cover'):
            m.verify_audit(audit, 100)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.directory = self.root/'run'; self.directory.mkdir()
        self.job = {'id': 'lstm--Cell_001963--r3', 'method': 'lstm', 'fit_end': 6132}
        self.plan = {'jobs': [self.job], 'common_origins': [1]}
        self.amendment = {'output_directory': 'run', 'base_plan_sha256': 'a'*64,
            'historical_attempts': {}, 'historical_completed_jobs': [], 'audited_completed_lstm_jobs': [],
            'diagnosed_failure_attempt': 'jobs/lstm--Cell_001963--r3/attempt-0001',
            'diagnosed_failure_job': self.job['id'], 'diagnosed_model_state_sha256': 'c'*64}
        self.identity = {'test': 'published'}
        self.patches = [patch.object(c, 'ROOT', self.root),
                        patch.object(c, 'amendment_identity', return_value=self.identity)]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def failure(self, historical=True, message='numerical assertion'):
        a, _ = begin_attempt(self.directory, self.job, 'a'*64)
        write_json_atomic(a/'failure.json', {'type': 'RuntimeError', 'message': message, 'automatic_retry': False})
        if historical:
            self.amendment['historical_attempts'][str(a.relative_to(self.directory))] = c.inventory_digest(a)
        else:
            write_json_atomic(a/'validation-amendment.json', self.identity)
        return a

    def test_only_exact_sealed_historical_numerical_failure_admitted(self):
        a = self.failure()
        state = c.snapshot(self.directory, self.plan, self.amendment, 'b'*64)
        self.assertEqual(len(state['failures']), 1)
        (a/'extra').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'Historical evidence changed'):
            c.snapshot(self.directory, self.plan, self.amendment, 'b'*64)

    def test_new_numerical_failure_never_auto_admitted(self):
        self.failure(historical=False)
        with self.assertRaisesRegex(ValueError, 'Unrecognized failure'):
            c.snapshot(self.directory, self.plan, self.amendment, 'b'*64)

    def test_missing_history_or_incomplete_attempt_rejected(self):
        self.amendment['historical_attempts']['jobs/missing/attempt-0001'] = 'e'*64
        with self.assertRaisesRegex(ValueError, 'missing'):
            c.snapshot(self.directory, self.plan, self.amendment, 'b'*64)
        self.amendment['historical_attempts'] = {}
        a, _ = begin_attempt(self.directory, self.job, 'a'*64)
        write_json_atomic(a/'validation-amendment.json', self.identity)
        with self.assertRaisesRegex(ValueError, 'Incomplete/active'):
            c.snapshot(self.directory, self.plan, self.amendment, 'b'*64)

    def test_attempt_identity_required(self):
        self.failure(historical=False)
        a = self.directory/'jobs'/self.job['id']/'attempt-0001'
        (a/'validation-amendment.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'outside this continuation'):
            c.snapshot(self.directory, self.plan, self.amendment, 'b'*64)

    def test_new_prediction_rule_and_same_seed_fingerprint_required(self):
        a = self.directory/'jobs'/self.job['id']/'attempt-0002'; a.mkdir(parents=True)
        write_json_atomic(a/'validation-amendment.json', self.identity)
        (a/'model/single').mkdir(parents=True)
        write_json_atomic(a/'model/single/model.json', {'scaler': {'mean': 0., 'sd': 1., 'fit_end': 6132}})
        np.savez(a/'reference.npz', targets=np.array([1]), raw=np.array([1.]))
        report = {'fit': {'model_state_sha256': 'c'*64}, 'predict': {
            'validation_amendment': self.identity,
            'timings': {'numerical_validation': m.check_origins([1.], [1.], [1.], 0., 1.)}}}
        with patch.object(c, 'validate_result', return_value=report):
            c.validate_amended_result(a, self.job, self.plan, 'a'*64, self.amendment, 'b'*64)
            report['predict']['timings']['numerical_validation']['mean'] = 10.
            with self.assertRaisesRegex(ValueError, 'different fitting scaler'):
                c.validate_amended_result(a, self.job, self.plan, 'a'*64, self.amendment, 'b'*64)
            report['predict']['timings']['numerical_validation'] = m.check_origins([2.], [2.], [2.], 0., 1.)
            with self.assertRaises(AssertionError):
                c.validate_amended_result(a, self.job, self.plan, 'a'*64, self.amendment, 'b'*64)
            report['predict']['timings']['numerical_validation'] = m.check_origins([1.], [1.], [1.], 0., 1.)
            report['fit']['model_state_sha256'] = 'd'*64
            with self.assertRaisesRegex(ValueError, 'same-seed'):
                c.validate_amended_result(a, self.job, self.plan, 'a'*64, self.amendment, 'b'*64)

    def test_runner_retains_failure_then_completes_fresh_attempt(self):
        self.amendment.update(recovery_limits={'maximum_attempts_per_job': 5,
            'maximum_failed_attempts_total': 12, 'cooldown_seconds': 0}, historical_completed_jobs=[])
        state = {'complete': [], 'failures': [], 'counts': {self.job['id']: 0}, 'next_job': self.job['id']}
        final = {'complete': [self.job['id']], 'failures': ['first'], 'counts': {self.job['id']: 2}, 'next_job': None}
        def launch(directory, job, attempt, phase, lock):
            (attempt/(phase+'.log')).write_text('test worker output')
            if attempt.name == 'attempt-0001':
                return 1
            write_json_atomic(attempt/(phase+'.json'), {'fake': True})
            return 0
        def summary(directory):
            write_json_atomic(directory/'summary.json', {'status': 'passed'})
            return {'status': 'passed'}
        with patch.object(c, 'verify_amendment', return_value=(self.amendment, 'b'*64)), \
             patch.object(c, 'verify_plan', return_value=(self.plan, {}, 'a'*64)), \
             patch.object(c, 'snapshot', side_effect=[state, final]), \
             patch.object(c, 'wait_for_quiet'), patch.object(c, 'launch', side_effect=launch), \
             patch.object(c, 'validate_amended_result'), \
             patch.object(c, 'readiness_failure', return_value=True), \
             patch.object(c, 'summarize', side_effect=summary):
            c.run(self.directory)
        first = self.directory/'jobs'/self.job['id']/'attempt-0001'
        second = first.parent/'attempt-0002'
        self.assertTrue((first/'failure.json').exists())
        self.assertFalse((first/'complete.json').exists())
        self.assertTrue((second/'complete.json').exists())
        self.assertTrue((self.directory/'validation-summary.json').exists())

    def test_runner_stops_on_unknown_error(self):
        self.amendment['recovery_limits'] = {'maximum_attempts_per_job': 5,
            'maximum_failed_attempts_total': 12, 'cooldown_seconds': 0}
        state = {'complete': [], 'failures': [], 'counts': {self.job['id']: 0}, 'next_job': self.job['id']}
        with patch.object(c, 'verify_amendment', return_value=(self.amendment, 'b'*64)), \
             patch.object(c, 'verify_plan', return_value=(self.plan, {}, 'a'*64)), \
             patch.object(c, 'snapshot', return_value=state), patch.object(c, 'wait_for_quiet'), \
             patch.object(c, 'launch', return_value=1), \
             patch.object(c, 'readiness_failure', return_value=False), \
             patch.object(c, 'summarize') as summary:
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                c.run(self.directory)
        summary.assert_not_called()
        self.assertEqual(len(list((self.directory/'jobs'/self.job['id']).iterdir())), 1)

    def test_launch_routes_only_lstm_prediction_to_new_worker(self):
        a = self.root/'attempt'; a.mkdir()
        child = Mock(); child.wait.return_value = 0
        with patch.object(c.subprocess, 'Popen', return_value=child) as popen:
            c.launch(self.directory, self.job, a, 'predict', 9)
            self.assertIn('validation_revision.worker', popen.call_args.args[0])
            c.launch(self.directory, self.job, a, 'fit', 9)
            self.assertIn('ensemble_study.evaluation_run', popen.call_args.args[0])
            self.assertEqual(popen.call_args.kwargs['pass_fds'], (9,))

    def test_interruption_terminates_worker(self):
        a = self.root/'attempt'; a.mkdir()
        child = Mock(); child.wait.side_effect = [KeyboardInterrupt(), 0]; child.poll.return_value = None
        with patch.object(c.subprocess, 'Popen', return_value=child):
            with self.assertRaises(KeyboardInterrupt):
                c.launch(self.directory, self.job, a, 'fit', 9)
        child.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
