"""Orchestration tests: all three real staging calls precede the first mocked Train."""
import tempfile, unittest
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from main_study.common import write,read
from main_study.campaign import execute_block
from test_attempt_staging import make_template

class BlockProbeOrderTests(unittest.TestCase):
    def run_case(self,fail_mode=None):
        calls=[]
        with tempfile.TemporaryDirectory() as t, ExitStack() as stack:
            c=Path(t);write(c/'origins.json',{'accepted_run':str(c/'accepted')})
            rows=[{'block_id':'block','configuration':m,'condition':'reference','planned_run_id':'job-'+m} for m in 'ABC']
            for r in rows:make_template(c/'jobs'/r['planned_run_id'],r['configuration'])
            progress={'attempts':{},'completed_blocks':[],'main_training_started':False}
            def command(args,log,*unused,**kwargs):
                job=Path(args[-2]);action=args[-1];mode=read(job/'job.json')['configuration'];calls.append((mode,action))
                if action=='Probe':
                    (job/'bundle_preflight/reference').mkdir(parents=True,exist_ok=False)
                    if mode==fail_mode:raise ValueError('fixture native probe failure')
                    write(job/'reports/job_preflight.json',{'status':'PASS'})
                else:
                    write(job/'reports/cell_complete.json',{'status':'PASS'})
                    write(job/'calibration.json',{'fixture':True})
            for name,value in [('plan',lambda:rows),('verify_campaign',lambda *a:{}),('environment',lambda *a:{}),
                               ('original_snapshot',lambda *a:{}),('command',command),('publish',lambda *a:None),
                               ('validate_finished_job',lambda job:{'status':'PASS','initial_state_digest':'same','configuration':job.name})]:
                stack.enter_context(patch('main_study.campaign.'+name,side_effect=value))
            stack.enter_context(patch('main_study.campaign.shutil.disk_usage',return_value=type('Disk',(),{'free':20*1024**3})()))
            result=execute_block(c,'block',progress,1)
            entry=read(c/'attempts/block/attempt_1/reports/block_status.json')
            return result,calls,entry,progress
    def test_all_probes_before_A_training(self):
        rc,calls,entry,p=self.run_case();self.assertEqual(rc,0)
        self.assertEqual(calls,[(m,'Probe') for m in 'ABC']+[(m,'Train') for m in 'ABC'])
        self.assertEqual(entry['status'],'COMPLETE');self.assertEqual(len(entry['completed_cells']),3)
    def test_B_probe_failure_does_not_train_A(self):
        rc,calls,entry,p=self.run_case('B');self.assertEqual(rc,1)
        self.assertEqual(calls,[('A','Probe'),('B','Probe')]);self.assertEqual(entry['completed_cells'],[])
        self.assertEqual(p['completed_blocks'],[])
    def test_C_probe_failure_does_not_train_A_or_B(self):
        rc,calls,entry,p=self.run_case('C');self.assertEqual(rc,1)
        self.assertEqual(calls,[(m,'Probe') for m in 'ABC']);self.assertEqual(entry['completed_cells'],[])
    def test_A_probe_failure_does_not_train(self):
        rc,calls,entry,p=self.run_case('A');self.assertEqual(rc,1)
        self.assertEqual(calls,[('A','Probe')]);self.assertFalse(any(a=='Train' for _,a in calls))
