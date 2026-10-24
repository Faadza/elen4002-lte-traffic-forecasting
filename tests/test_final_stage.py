"""Final-only provenance, repetition and resume gates on invented evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from final_stage import run as f
from ensemble_study.evaluation_plan import jobs_for
from tools.development_store import write_json_atomic, load_success


class FinalStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.directory = self.root/'final'; self.directory.mkdir()
        self.config_path = self.root/'config.json'; self.config_path.write_text('{}')
        self.receipt_path = self.root/'receipt.json'; self.receipt_path.write_text('{}')
        self.patches = [patch.object(f, 'ROOT', self.root), patch.object(f, 'CONFIG', self.config_path),
            patch.object(f, 'RECEIPT', self.receipt_path),
            patch.object(f, 'amendment_identity', return_value={'test': 'published'})]
        for item in self.patches: item.start()

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    def fake_plan(self):
        jobs = [{'predict': i < 741} for i in range(1539)]
        base = {'period': 'final', 'primary_model': 'ensemble_l1_r0p1',
            'primary_comparator': 'lightgbm_l1', 'selection_sha256': 'a'*64,
            'confirmation_freeze': {'sha256': 'b'*64}, 'jobs': jobs}
        config = {k:base[k] for k in ('primary_model','primary_comparator','selection_sha256')}
        config.update(confirmation_sha256={'summary.json':'b'*64,'validation-summary.json':'c'*64},
            recovery_limits={'maximum_attempts_per_job':5,'maximum_failed_attempts_total':12,
                             'cooldown_seconds':0})
        write_json_atomic(self.directory/'plan.json', base)
        write_json_atomic(self.directory/'cost-plan.json', {'formal':True})
        return base,config

    def test_final_validation_plan_binds_both_summaries_and_cost_plan(self):
        base,config = self.fake_plan()
        plan = f.validation_plan(self.directory,base,config)
        self.assertEqual(plan['confirmation_sha256'],config['confirmation_sha256'])
        self.assertEqual(plan['cost_plan_sha256'],f.sha256(self.directory/'cost-plan.json'))
        self.assertFalse(plan['settings_retuned'])
        self.assertTrue(plan['independent_final_fits'])
        self.assertEqual(plan['historical_attempts'],{})

    def test_wrong_period_comparator_selection_or_job_count_rejected(self):
        base,config = self.fake_plan()
        for key,value in [('period','confirmation'),('primary_model','ensemble_l1_r0p5'),
            ('primary_comparator','ridge'),('selection_sha256','d'*64),('jobs',base['jobs'][:-1]),
            ('confirmation_freeze',{'sha256':'d'*64})]:
            changed=copy.deepcopy(base);changed[key]=value
            with self.assertRaises(ValueError):
                f.validation_plan(self.directory,changed,config)
        changed=copy.deepcopy(base);changed['jobs'][0]['predict']=False
        with self.assertRaises(ValueError): f.validation_plan(self.directory,changed,config)

    def test_final_validation_plan_cannot_be_missing_or_changed(self):
        base,config = self.fake_plan();config['output_directory']='final'
        with patch.object(f,'verify_publication',return_value=config):
            with self.assertRaises(FileNotFoundError): f.verify_amendment(self.directory)
            expected=f.validation_plan(self.directory,base,config)
            write_json_atomic(self.directory/'validation-plan.json',expected)
            self.assertEqual(f.verify_amendment(self.directory)[0],expected)
            (self.directory/'cost-plan.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError): f.verify_amendment(self.directory)

    def test_unpublished_confirmation_stops_before_data_preparation(self):
        with patch.object(f,'verify_publication',side_effect=ValueError('Unverified confirmation')), \
             patch('ensemble_study.evaluation_plan.prepare') as prepare:
            with self.assertRaisesRegex(ValueError,'Unverified confirmation'):
                f.prepare_final(operator='user',host_ready=True)
        prepare.assert_not_called()

    def test_both_audits_must_be_passed_and_in_publication(self):
        names=['docs/l1-l2-confirmation-results/arithmetic-audit.json',
               'docs/l1-l2-confirmation-results/integrity-audit.json']
        hashes={}
        for name in names:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            write_json_atomic(path,{'status':'passed'});hashes[name]=f.sha256(path)
        with self.assertRaisesRegex(ValueError,'Both independent'):
            f.verify_confirmation_audits({'independent_audits_sha256':{}},{'payload_sha256':{}})
        with self.assertRaisesRegex(ValueError,'Published confirmation'):
            f.verify_confirmation_audits({'independent_audits_sha256':hashes},{'payload_sha256':{}})
        f.verify_confirmation_audits({'independent_audits_sha256':hashes},{'payload_sha256':hashes})
        (self.root/names[0]).write_text('{"status":"failed"}')
        hashes[names[0]]=f.sha256(self.root/names[0])
        with self.assertRaisesRegex(ValueError,'Published confirmation'):
            f.verify_confirmation_audits({'independent_audits_sha256':hashes},{'payload_sha256':hashes})

    def test_user_execution_requires_host_assertion(self):
        with self.assertRaises(ValueError): f.prepare_final(operator='user',host_ready=False)

    def test_unprepared_check_does_not_create_output(self):
        target=self.root/'not_prepared'
        self.config_path.write_text(json.dumps({'output_directory':'not_prepared'}))
        with patch.object(f,'verify_publication',return_value={}), patch.object(f.sys,'argv',['runner','--check']):
            f.main()
        self.assertFalse(target.exists())

    def test_unprepared_run_does_not_create_output(self):
        target=self.root/'not_prepared'
        with self.assertRaisesRegex(ValueError,'both final plans'):
            f.run(target,operator='user',host_ready=True)
        self.assertFalse(target.exists())

    def test_final_1539_jobs_have_741_predictions_and_no_extra_lstm_repeats(self):
        from ensemble_study.evaluation_plan import expected_conditions
        selection={'conditions': expected_conditions(
            {'l1': {'trees': 512, 'leaves': 31}, 'l2': {'trees': 128, 'leaves': 7}},
            {'l1': .1, 'l2': .2})}
        jobs=jobs_for('final',[f'Cell_{i}' for i in range(57)],selection['conditions'])
        self.assertEqual(len(jobs),1539);self.assertEqual(sum(x['predict'] for x in jobs),741)
        neural=[j for j in jobs if j['method']=='lstm']
        self.assertEqual(len(neural),171)
        self.assertEqual({j['seed'] for j in neural},{2026,2027,2028})
        self.assertTrue(all(j['predict'] for j in neural))
        self.assertTrue(all(not j['predict'] for j in jobs if j['method']!='lstm' and j['repeat']>1))

    def workflow(self, *, mismatch=False, readiness_failure=False, unknown_failure=False):
        (self.directory/'plan.json').write_text('{}')
        (self.directory/'validation-plan.json').write_text('{}')
        jobs=[{'id':f'ridge--cell--r{i}','method':'ridge','repeat':i,'predict':i==1} for i in (1,2,3)]
        plan={'jobs':jobs,'common_origins':[]}
        amendment={'output_directory':'final','base_plan_sha256':'a'*64,'historical_attempts':{},
            'historical_completed_jobs':[], 'diagnosed_failure_job':None,
            'recovery_limits':{'maximum_attempts_per_job':5,'maximum_failed_attempts_total':12,'cooldown_seconds':0}}
        state={'complete':[],'counts':{j['id']:0 for j in jobs},'failures':[],'next_job':jobs[0]['id']}
        phases=[]
        def launch(directory,job,attempt,phase,lock):
            phases.append((job['repeat'],attempt.name,phase))
            if (readiness_failure or unknown_failure) and len(phases)==1: return 1
            write_json_atomic(attempt/(phase+'.json'),{'model_state_sha256':'same' if not mismatch or job['repeat']==1 else 'changed'})
            return 0
        def result(attempt,*args,**kwargs):
            return {'fit':json.loads((attempt/'fit.json').read_text())}
        def summary(directory):
            write_json_atomic(directory/'summary.json',{'status':'passed'})
            return {'status':'passed'}
        final={'complete':[j['id'] for j in jobs],'failures':['retained'] if readiness_failure else []}
        with patch.object(f,'verify_amendment',return_value=(amendment,'b'*64)), \
             patch.object(f,'verify_plan',return_value=(plan,{},'a'*64)), \
             patch.object(f,'snapshot',side_effect=[state,final]), patch.object(f,'wait_for_quiet'), \
             patch.object(f,'launch',side_effect=launch),patch.object(f,'validate_amended_result',side_effect=result), \
             patch.object(f,'readiness_failure',return_value=readiness_failure), \
             patch.object(f,'summarize',side_effect=summary):
            f.run(self.directory,operator='user',host_ready=True)
        return phases

    def test_deterministic_repeats_are_fit_only_and_counted(self):
        phases=self.workflow()
        self.assertEqual([(r,p) for r,a,p in phases],[(1,'fit'),(1,'predict'),(2,'fit'),(3,'fit')])
        self.assertTrue((self.directory/'validation-summary.json').exists())

    def test_changed_deterministic_state_stops_and_keeps_failure(self):
        with self.assertRaisesRegex(ValueError,'Repeated deterministic fit'):
            self.workflow(mismatch=True)
        self.assertTrue((self.directory/'jobs/ridge--cell--r2/attempt-0001/failure.json').exists())
        self.assertFalse((self.directory/'summary.json').exists())

    def test_readiness_failure_retried_without_losing_attempt(self):
        phases=self.workflow(readiness_failure=True)
        self.assertEqual(phases[0],(1,'attempt-0001','fit'))
        self.assertEqual(phases[1],(1,'attempt-0002','fit'))
        self.assertTrue((self.directory/'jobs/ridge--cell--r1/attempt-0001/failure.json').exists())

    def test_unknown_failure_stops_without_summary(self):
        with self.assertRaisesRegex(RuntimeError,'worker failed'):
            self.workflow(unknown_failure=True)
        self.assertFalse((self.directory/'summary.json').exists())


if __name__=='__main__': unittest.main()
