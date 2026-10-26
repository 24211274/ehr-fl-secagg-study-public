"""Full evaluator file workflow on wholly fabricated audit-event fixtures only."""
import contextlib,csv,io,tempfile,unittest,shutil
from pathlib import Path
from unittest.mock import patch
import torch
from main_study.common import ROOT,CLIENTS,SEEDS,read,write,sha
from main_study.evaluation import calibrate_validation
from main_study.final_evaluator import evaluate
from ehr_fedavg.task import model_factory,setup_torch
from ehr_preprocessing.pipeline import CLIENTS as HOSPITALS


class FabricatedSource:
    calls=[]
    def __init__(self,*args):
        schema=read(ROOT/'config/encoding_schema.json');self.tables={}
        labels=[];events=[]
        for part in ('test_known','test_held_out'):
            for c in CLIENTS:
                for y in (0,1):
                    sid=f'fabricated-{part}-{c}-{y}'
                    labels.append({'session_id':sid,'hospital_id':HOSPITALS[c],'partition':part,'binary_label':str(y),'scenario_id':'numeric_fixture','primary_class':'fixture'})
                    r={'session_id':sid,'hospital_id':HOSPITALS[c],'event_id':sid+'-event','sequence_position':'0','timestamp_utc':'2025-01-01T10:00:00Z'}
                    for f in schema['categorical_fields']:r[f['name']]=next(v for v in f['categories'] if not v.startswith('__'))
                    for f in schema['binary_fields']:r[f]='False'
                    events.append(r)
        for name,rows in [('events.csv',events),('labels.csv',labels)]:
            s=io.StringIO();w=csv.DictWriter(s,list(rows[0]));w.writeheader();w.writerows(rows);self.tables[name]=s.getvalue()
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def verify(self):return [{'fixture_only':True}]
    @contextlib.contextmanager
    def text(self,name):
        self.calls.append(name);yield io.StringIO(self.tables[name])


class EvaluatorWholeFlowTests(unittest.TestCase):
    def test_authorised_fake_30_models_two_tests_end_to_end(self):
        setup_torch()
        with tempfile.TemporaryDirectory() as td:
            campaign=Path(td);data=campaign/'frozen/development';data.mkdir(parents=True)
            shutil.copyfile(ROOT/'config/encoding_schema.json',data/'encoding_schema.json')
            shutil.copyfile(ROOT/'protocol/decisions/evidence/S26_relative_time_scaler.json',data/'relative_time_scaler.json')
            write(campaign/'campaign_manifest.json',{'fixture':True});write(campaign/'origins.json',{'accepted_run':str(campaign/'unused')})
            # One dummy checkpoint is reused deliberately: this test exercises evaluator
            # mechanics, not main-run seeds, optimisation or detection performance.
            cp=campaign/'fixture.pt';torch.save(model_factory().state_dict(),cp)
            rows=[{'session_id':f'val{c}{i}','client':c,'partition':'validation','loss':float(i)} for c in CLIENTS for i in range(10)]
            cal=calibrate_validation(rows,{r['session_id'] for r in rows});cal['selected_checkpoint_sha256']=sha(cp)
            write(campaign/'cal.json',cal)
            models=[{'configuration':m,'condition':c,'seed':s,'status':'PASS','rounds':30,'test_data_opened':False,
                'checkpoint':'fixture.pt','calibration':'cal.json','selected_checkpoint_sha256':sha(cp),'calibration_sha256':sha(campaign/'cal.json')}
                for m in 'ABC' for c in ('reference','noniid') for s in SEEDS]
            write(campaign/'selected_models_manifest.json',{'models':models,'campaign_manifest_sha256':sha(campaign/'campaign_manifest.json'),'package_manifest_sha256':'f'*64})
            receipt=campaign/'approval.json';write(receipt,{'approved':True,'action':'EVALUATE_FINAL_TESTS',
                'campaign_manifest_sha256':sha(campaign/'campaign_manifest.json'),'package_manifest_sha256':'f'*64,
                'selected_models_manifest_sha256':sha(campaign/'selected_models_manifest.json')})
            FabricatedSource.calls=[]
            with patch('main_study.final_evaluator.verify_campaign'),patch('main_study.final_evaluator.package_integrity',return_value='f'*64),patch('main_study.common.package_integrity',return_value='f'*64),patch('main_study.final_evaluator.environment',return_value={'fixture':True}),patch('main_study.final_evaluator.original_snapshot',return_value={'fixture':True}),patch('main_study.final_evaluator.BenchmarkSource',FabricatedSource),patch('main_study.final_evaluator.publish') as publish:
                rc=evaluate(campaign,receipt,campaign/'not_a_real_dataset')
            self.assertEqual(rc,0);self.assertEqual(FabricatedSource.calls,['labels.csv','events.csv'])
            summary=read(campaign/'final_evaluation/reports/paired_seed_analysis.json');self.assertEqual(summary['cells'],60)
            self.assertEqual(summary['status'],'COMPLETE');self.assertEqual(len(list((campaign/'final_evaluation/reports/cells').glob('*.json'))),60)
            self.assertEqual(len(list((campaign/'final_evaluation/LOCAL_ONLY').glob('*.csv'))),60)
            self.assertTrue(read(campaign/'final_evaluation/reports/stage_status.json')['test_metrics_computed'])
            publish.assert_called_once()
