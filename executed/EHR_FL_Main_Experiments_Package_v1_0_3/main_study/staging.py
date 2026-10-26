"""Copy sealed job inputs, never the output of a previous FAB probe.

This is execution staging only. It does not change a model, data membership,
training schedule, checkpoint rule, protocol primitive, or measurement scope.
"""
from __future__ import annotations
import shutil, uuid, tomllib
from pathlib import Path
from .common import read, write

INPUTS = ('apps', 'client_data', 'runtime.json', 'job.json')
PREPARATION_OUTPUTS = {'reports', 'bundle_preflight', 'probe_flower_home', '__pycache__'}


def stage_attempt_job(template: Path, job: Path) -> Path:
    """Create a fresh relocated attempt from the four sealed input roots.

    Existing destinations and trained templates are refused. Probe directories
    remain intact in the template and are not copied. In particular, the bundle
    checker can retain its exist_ok=False protection for its new work directory.
    """
    template, job = Path(template), Path(job)
    if not template.is_dir() or template.is_symlink():
        raise ValueError('Expected a regular prepared job directory')
    if job.exists() or job.is_symlink():
        raise FileExistsError('A fresh attempt destination is required: '+str(job))
    if job.resolve().is_relative_to(template.resolve()):
        raise ValueError('Attempt cannot be created inside its template')
    extra = {p.name for p in template.iterdir()} - set(INPUTS) - PREPARATION_OUTPUTS
    if extra:
        raise ValueError('Unexpected/trained job artefacts in template: '+repr(sorted(extra)))
    for name in INPUTS:
        p = template/name
        if p.is_symlink() or not (p.is_dir() if name in ('apps','client_data') else p.is_file()):
            raise ValueError('Missing/unsafe sealed job input: '+name)
        if p.is_dir() and any(x.is_symlink() for x in p.rglob('*')):
            raise ValueError('Symlink within sealed job input: '+name)
    meta = read(template/'job.json')
    co, mode, rounds = meta['condition'], meta['configuration'], meta['rounds']
    if co not in ('reference','noniid') or mode not in ('A','B','C') or type(rounds) is not int or not 1 <= rounds <= 30:
        raise ValueError('Invalid prepared job identity')
    conf = read(template/'apps'/co/'ehr_fedavg/fixed_config.json')
    if conf['rounds'] != rounds or conf['conditions'] != [co] or conf['training_seed'] != meta['training_seed']:
        raise ValueError('Prepared job configuration is inconsistent')
    root = template.resolve()
    def ignore(directory, names):
        if Path(directory).resolve() == root:
            return set(names)-set(INPUTS)
        return {n for n in names if n == '__pycache__' or n.endswith(('.pyc','.pyo','.fab'))}
    shutil.copytree(template, job, ignore=ignore)
    runtime = read(job/'runtime.json')
    runtime['run_token'] = uuid.uuid4().hex
    write(job/'runtime.json', runtime)
    # Same writer and replacements as the original main-block runner.
    from ehr_fedavg.host import write_project
    project = job/'apps'/co/'pyproject.toml'
    write_project(project, job, co, conf['ports']['control'])
    text = project.read_text().replace('num-server-rounds = 6', f'num-server-rounds = {rounds}')
    if mode == 'C':
        text = text.replace('ehr_fedavg.server_app:app','ehr_secagg.server_app:app').replace('ehr_fedavg.client_app:app','ehr_secagg.client_app:app')
    if tomllib.loads(text)['tool']['flwr']['app']['config']['num-server-rounds'] != rounds:
        raise ValueError('Relocated round configuration is inconsistent')
    project.write_text(text)
    if any((job/n).exists() for n in PREPARATION_OUTPUTS):
        raise ValueError('Preparation outputs were copied into the new attempt')
    return job
