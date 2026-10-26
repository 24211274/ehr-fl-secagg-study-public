"""Client-local C training, same numerical kernel as B; no update diagnostics exported."""
from __future__ import annotations
from pilot_core.core import derived_seed
from central_smoke.identity import write_json
from ehr_fedavg.task import LocalTask,cfg,fit_one_client
from ehr_fedavg.math_checks import arrays_from_model
from .policy import local_range_check
from mai_measurement import client_span

class SecAggLocalTask(LocalTask):
    def fit(self,parameters,config):
        before=self.validate(parameters,config);r=int(config['server_round'])
        if r not in range(1,cfg()['rounds']+1):raise ValueError('Wrong training round')
        record=self.output/f'fit_round_{r:02d}.json'
        if record.exists():raise ValueError('Repeated local fit; no automatic retry')
        rows=sorted([d for d in self.draws if d['condition']==self.condition and int(d['outer_epoch'])==r],key=lambda d:int(d['client_draw_position']))
        ids=[d['session_id'] for d in rows]
        seed=derived_seed(cfg()['training_seed'],'client-fit',self.client,r)
        with client_span('local_fit_with_checks'):
            history,metrics=fit_one_client(self.model,self.data['training'],ids,seed)
        arrays=arrays_from_model(self.model);local_range_check(arrays)
        # Fit traces stay local; omit losses, gradients and output fingerprints even here.
        safe={'status':'PASS','client':self.client,'condition':self.condition,'round':r,
              'input_digest':before,'steps':metrics['steps'],'draws':len(ids),'fresh_optimizer':True,
              'training_seed':seed,'range_check_passed':True,
              'batch_sizes':[x['sessions'] for x in history],
              'fit_wall_seconds':metrics['fit_wall_seconds'],'fit_cpu_seconds':metrics['fit_cpu_seconds'],
              'test_data_opened':False,'local_only':True}
        write_json(record,safe)
        return arrays,len(ids),{'client':self.client,'input_digest':before,'steps':metrics['steps'],'draws':len(ids)}


