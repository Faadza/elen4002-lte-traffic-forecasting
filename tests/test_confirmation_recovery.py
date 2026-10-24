"""Invented files/processes only: recovery must not accept or hide bad work."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("confirmation_recovery", ROOT/"docs/commands/confirmation_recovery.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
from tools.development_store import begin_attempt, complete_attempt


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.plan = {"jobs": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
        self.digest = "a"*64
        self.config = {"maximum_attempts_per_job": 5, "maximum_failed_attempts_total": 12,
                       "maximum_quiet_wait_seconds": 5, "maximum_cpu_percent": 20, "poll_seconds": 1}

    def attempt(self, index=0, *, success=False, failure=False, phase="predict", side="before"):
        job = self.plan["jobs"][index]
        a, _ = begin_attempt(self.root, job, self.digest, resume=True, retry_failed=True)
        if success:
            (a/"result.json").write_text('{}\n')
            complete_attempt(a, job, self.digest)
        elif failure:
            (a/"failure.json").write_text(json.dumps({"type": "RuntimeError", "automatic_retry": False,
                "message": f"{phase} worker failed (1); inspect {a/(phase+'.log')}"}))
            (a/(phase+'.log')).write_text('Traceback (most recent call last):\n'
                f'  readiness_{side} = native_readiness(identity)\n'
                '  File "/example/ensemble_study/common.py", line 98, in native_readiness\n'
                '  File "/example/ensemble_study/common.py", line 76, in verify_saved_readiness\n'
                +recovery.BACKGROUND_ERROR+'\n')
        return a

    def state(self):
        return recovery.snapshot(self.root, self.plan, self.digest)

    def test_recognizes_each_phase_and_boundary(self):
        for phase in ("fit", "predict"):
            for side in ("before", "after"):
                with self.subTest(phase=phase, side=side):
                    a=self.attempt(failure=True, phase=phase, side=side)
                    self.assertTrue(recovery.readiness_failure(a))

    def test_rejects_unknown_log_parent_message_and_interruption(self):
        a=self.attempt(failure=True)
        for text in ["ValueError: different error", recovery.BACKGROUND_ERROR,
                     (a/"predict.log").read_text()+"KeyboardInterrupt\n"]:
            with self.subTest(text=text):
                with patch.object(Path, "read_text", return_value=text):
                    # Only replace log reads, not JSON metadata.
                    with patch.object(recovery, "_read_json", return_value={"type":"RuntimeError",
                        "automatic_retry":False, "message":f"predict worker failed (1); inspect {a/'predict.log'}"}):
                        self.assertFalse(recovery.readiness_failure(a))
        data=json.loads((a/"failure.json").read_text());data['type']='KeyboardInterrupt'
        (a/"failure.json").write_text(json.dumps(data))
        self.assertFalse(recovery.readiness_failure(a))

    def test_preserves_success_and_retries_only_new_failure(self):
        self.attempt(success=True)
        self.attempt(1,failure=True)
        before=self.state()
        self.attempt(1,success=True)
        self.attempt(2,failure=True)
        after=self.state()
        self.assertEqual(len(after['complete']),2)
        self.assertEqual(recovery.verify_transition(before,after,1),'jobs/c/attempt-0001')

    def test_stale_failure_is_not_retry_evidence(self):
        self.attempt(failure=True);state=self.state()
        with self.assertRaises(ValueError):recovery.verify_transition(state,state,1)

    def test_new_unknown_failure_blocks_even_with_old_known_error(self):
        self.attempt(failure=True);self.state()
        a=self.attempt(failure=True)
        (a/'predict.log').write_text('ValueError: model changed\n')
        with self.assertRaises(ValueError):self.state()

    def test_multiple_new_failures_are_rejected(self):
        before=self.state();self.attempt(failure=True);self.attempt(failure=True)
        with self.assertRaises(ValueError):recovery.verify_transition(before,self.state(),1)

    def test_sigint_exit_is_not_retried_after_known_failure(self):
        before=self.state();self.attempt(failure=True)
        for code in (-2,130,2):
            with self.subTest(code=code), self.assertRaises(ValueError):
                recovery.verify_transition(before,self.state(),code)

    def test_original_failure_tampering_detected(self):
        a=self.attempt(failure=True);before=self.state()
        (a/'new.txt').write_text('changed historical attempt')
        self.attempt(failure=True)
        with self.assertRaises(ValueError):recovery.verify_transition(before,self.state(),1)

    def test_success_tampering_and_incomplete_attempt_are_rejected(self):
        a=self.attempt(success=True);(a/'result.json').write_text('changed')
        with self.assertRaises(ValueError):self.state()

    def test_incomplete_attempt_blocks(self):
        self.attempt()
        with self.assertRaises(ValueError):self.state()

    def test_nonprefix_success_blocks(self):
        self.attempt(1,success=True)
        with self.assertRaises(ValueError):self.state()

    def test_job_and_global_caps_survive_supervisor_restarts(self):
        for _ in range(4):self.attempt(failure=True)
        recovery.enforce_budget(self.state(),self.config)
        self.attempt(failure=True)
        with self.assertRaises(RuntimeError):recovery.enforce_budget(self.state(),self.config)
        # Global budget counts persisted failure directories, not process memory.
        config={**self.config,'maximum_attempts_per_job':20,'maximum_failed_attempts_total':5}
        with self.assertRaises(RuntimeError):recovery.enforce_budget(self.state(),config)

    def test_zero_budget_and_success_at_limit(self):
        config={**self.config,'maximum_attempts_per_job':0}
        with self.assertRaises(RuntimeError):recovery.enforce_budget(self.state(),config)
        for i in range(3):self.attempt(i,success=True)
        recovery.enforce_budget(self.state(),config)

    def test_success_requires_complete_panel(self):
        before=self.state();self.attempt(success=True)
        with self.assertRaises(ValueError):recovery.verify_transition(before,self.state(),0)
        for i in (1,2):self.attempt(i,success=True)
        self.assertIsNone(recovery.verify_transition(before,self.state(),0))

    def test_quiet_wait_records_actual_samples_then_returns(self):
        values=iter([55,25,20,19]);events=[];clock=[0]
        def sample(interval):clock[0]+=interval;return next(values)
        with redirect_stdout(io.StringIO()):
            recovery.wait_for_quiet(self.config,lambda e,**v:events.append(v),sample=sample,
                sleep=lambda t:clock.__setitem__(0,clock[0]+t),clock=lambda:clock[0])
        self.assertEqual([e['quiet'] for e in events],[False,True])
        self.assertEqual(events[-1]['samples_percent'],[20,19])

    def test_busy_timeout_and_invalid_measurements_stop(self):
        for value,kind in [(99,RuntimeError),(float('nan'),ValueError),(-1,ValueError),(True,ValueError)]:
            clock=[0]
            def sample(interval):clock[0]+=interval;return value
            with self.subTest(value=value),redirect_stdout(io.StringIO()),self.assertRaises(kind):
                recovery.wait_for_quiet(self.config,lambda *a,**k:None,sample=sample,
                    sleep=lambda t:clock.__setitem__(0,clock[0]+t),clock=lambda:clock[0])

    def test_existing_run_lock_blocks_overlap(self):
        with recovery.run_lock(self.root):
            with self.assertRaises(RuntimeError):
                with recovery.run_lock(self.root):pass

    def test_changed_plan_and_publication_fail_before_execution(self):
        with self.assertRaises(ValueError):recovery.verify_inputs({}, {'verified':False})
        config={'output_directory':str(self.root),'plan_sha256':'0'*64}
        (self.root/'plan.json').write_text('{}')
        files=['docs/commands/confirmation_recovery.py','docs/commands/confirmation-recovery-v1.json',
               'docs/commands/resume_ensemble_confirmation_mac.sh']
        fixtures={ROOT/files[1]:self.root/'recovery-config.json',
                  ROOT/files[2]:self.root/'resume.sh'}
        fixtures[ROOT/files[1]].write_text('{}\n')
        fixtures[ROOT/files[2]].write_text('# Synthetic launcher fixture; never executed.\n')
        original_sha256=recovery.sha256
        def fixture_sha256(path):
            return original_sha256(fixtures.get(Path(path),Path(path)))
        with patch.object(recovery,'sha256',side_effect=fixture_sha256):
            publication={'verified':True,'payload_sha256':{p:recovery.sha256(ROOT/p) for p in files}}
            with self.assertRaisesRegex(ValueError,'plan changed'):recovery.verify_inputs(config,publication)
            publication['payload_sha256'][files[0]]='0'*64
            with self.assertRaisesRegex(ValueError,'publication differs'):recovery.verify_inputs(config,publication)

    def test_launch_logging_failure_and_signals_clean_up_child(self):
        for error in (RuntimeError('log failure'), recovery.RecoveryInterrupted(15), KeyboardInterrupt()):
            child=MagicMock(pid=987654321);child.poll.return_value=None
            d=self.root/str(len(list(self.root.iterdir())));d.mkdir()
            with patch.object(recovery.subprocess,'Popen',return_value=child), \
                 patch.object(recovery.os,'killpg') as kill, self.assertRaises(type(error)):
                recovery.launch(['invented-runner'],d,lambda *a,**k: (_ for _ in ()).throw(error))
            kill.assert_called_once_with(child.pid,recovery.signal.SIGINT)
            child.wait.assert_called_once_with(timeout=5)
        for sig in (recovery.signal.SIGTERM,recovery.signal.SIGHUP):
            with self.assertRaises(recovery.RecoveryInterrupted):recovery.interrupt(sig,None)

    def test_recovery_retries_known_failure_then_finishes_once(self):
        config={**self.config,'plan_sha256':self.digest,'handoff_path':'invented.sh',
                'selection_sha256':'b'*64,'cooldown_seconds':0}
        session=self.root/'session';session.mkdir();events=[];calls=[]
        original_read=recovery._read_json
        def read(path):
            return {} if path==recovery.RECEIPT else original_read(path)
        def launch(command,logdir,emit):
            calls.append(command)
            session=self.root/'sessions'/f'session-{len(calls):04d}';session.mkdir(parents=True)
            (session/'started.json').write_text(json.dumps({'stage':'confirmation','plan_sha256':self.digest}))
            (session/'finished.json').write_text(json.dumps({'status':'failed_or_interrupted' if len(calls)==1 else 'passed'}))
            (session/'operator.json').write_text(json.dumps({'operator':'user','host_ready_assertion':True}))
            if len(calls)==1:
                a=self.attempt(failure=True)
                message=original_read(a/'failure.json')['message']
                (logdir/'runner.log').write_text(f'Traceback:\nRuntimeError: {message}\n')
                return 1
            for i in range(3):self.attempt(i,success=True)
            (self.root/'summary.json').write_text(json.dumps({'status':'passed',
                'plan_sha256':self.digest,'selection_sha256':'b'*64}))
            return 0
        with patch.object(recovery,'verify_inputs'),patch.object(recovery,'_read_json',side_effect=read), \
             patch.object(recovery,'wait_for_quiet'),patch.object(recovery,'launch',side_effect=launch), \
             patch.object(recovery.time,'sleep'),redirect_stdout(io.StringIO()):
            recovery.recover(config,self.root,self.plan,session,lambda e,**k:events.append(e))
        self.assertEqual(len(calls),2)
        self.assertEqual(events.count('external_retry_scheduled'),1)
        self.assertEqual(events.count('passed'),1)
        self.assertEqual(len(self.state()['complete']),3)
        self.assertEqual(len(self.state()['failures']),1)

    def test_concurrent_or_changed_runner_sessions_stop(self):
        before={'session-0001':{'files':{'started.json':'a'},'status':'failed_or_interrupted'}}
        after={**before,'session-0002':{'files':{'started.json':'b'},'status':'passed'}}
        self.assertEqual(recovery.verify_sessions(before,after,0),'session-0002')
        for bad in (before,{**after,'session-0003':{}},{**after,'session-0001':{'started.json':'changed'}}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):recovery.verify_sessions(before,bad,0)
        with self.assertRaisesRegex(ValueError,'outcome disagrees'):recovery.verify_sessions(before,after,1)

    def test_lock_contention_cannot_adopt_another_runners_failure(self):
        a=self.attempt(failure=True);log=self.root/'runner.log'
        log.write_text('RuntimeError: Another parent or worker still owns this run\n')
        with self.assertRaisesRegex(ValueError,'This invocation did not report'):
            recovery.verify_parent_failure(log,a)
        message=json.loads((a/'failure.json').read_text())['message']
        log.write_text(f'Traceback:\nRuntimeError: {message}\n')
        recovery.verify_parent_failure(log,a)


if __name__ == '__main__':
    unittest.main()
