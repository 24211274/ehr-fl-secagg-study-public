"""Test-only fresh-process probes for dynamically bound Flower App fixtures.

Do not import a fixture's application into the unittest process: that process
has already imported the release's default ehr_fedavg._embedded_resources.
Changing sys.path is insufficient to replace an existing sys.modules entry.
Production scripts/job_worker.py already isolates each job; this helper follows
that process boundary without changing any production checker or resource.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _check_arguments(app: Path, job: Path, mode: str, condition: str) -> None:
    if mode not in ('B', 'C') or condition not in ('reference', 'noniid'):
        raise ValueError('Unsupported test fixture configuration or condition')
    expected = job.resolve() / 'apps' / condition
    if app.resolve() != expected or not app.is_dir():
        raise ValueError('Fixture app must be the selected job/apps/condition')


def run_isolated_probe(app: Path, job: Path, mode: str, condition: str,
                       report: Path, *, native: bool = False,
                       timeout: float = 600) -> dict:
    """Use the current interpreter, not another environment or a shell command.

    The resource-only branch requires no Flower. The native branch uses the
    unmodified native FAB checker and must not silently fall back to a test ZIP.
    Both branches work only on temporary test fixtures, not benchmark data.
    """
    app, job, report = Path(app).resolve(), Path(job).resolve(), Path(report).resolve()
    _check_arguments(app, job, mode, condition)
    if report.exists():
        raise FileExistsError('Test probe report already exists: ' + str(report))
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()),
               '--app', str(app), '--job', str(job), '--mode', mode,
               '--condition', condition, '--report', str(report)]
    if native:
        command.append('--native')
    result = subprocess.run(command, cwd=job, env=env, capture_output=True,
                            text=True, timeout=timeout)
    audit = json.loads(report.read_text(encoding='utf-8')) if report.is_file() else {}
    if result.returncode != 0 or audit.get('status') != 'PASS':
        raise RuntimeError('Isolated test fixture probe failed: ' +
                           str(audit.get('message', 'no passing report')) +
                           '\n' + (result.stdout + result.stderr)[-10000:])
    if native and audit.get('native_flower_build_executed') is not True:
        raise RuntimeError('Native Flower build was not executed')
    origin = audit['test_fixture_import']
    if (origin['app_directory'] != str(app) or not origin['isolated_interpreter']
            or not origin['bytecode_writes_disabled']
            or origin['config_sha256'] != hashlib.sha256(
                (app / 'ehr_fedavg/fixed_config.json').read_bytes()).hexdigest()):
        raise RuntimeError('Test fixture process/path/resource identity differs')
    return audit


def _child(app: Path, job: Path, mode: str, condition: str,
           report: Path, native: bool) -> dict:
    _check_arguments(app, job, mode, condition)
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError('Fixture probe requires -I -B')
    # A new process must not inherit a previously loaded application resource.
    if any(n == 'ehr_fedavg' or n.startswith('ehr_fedavg.') or
           n == 'ehr_secagg' or n.startswith('ehr_secagg.') for n in sys.modules):
        raise RuntimeError('Application module imported before fixture path binding')
    # Never add ROOT/app as fallback: every application import must come from
    # this fixture. ROOT supplies support packages only, as in job_worker.py.
    sys.path[:0] = [str(app), str(ROOT), str(ROOT / 'validation_support')]
    res = importlib.import_module('ehr_fedavg.resources')
    carrier = importlib.import_module('ehr_fedavg._embedded_resources')
    for module in (res, carrier):
        if not Path(module.__file__).resolve().is_relative_to(app):
            raise RuntimeError('Application import outside selected fixture')
    verified = res.verify_source_resources(app)  # Strict production check, unchanged.
    conf = res.read_document('fixed_config.json')
    if conf['conditions'] != [condition]:
        raise ValueError('Fixture condition does not match its configuration')
    identity = {'app_directory': str(app), 'resources_module': str(res.__file__),
                'carrier_module': str(carrier.__file__),
                'config_sha256': res.resource_sha256('fixed_config.json'),
                'schema_sha256': res.resource_sha256('encoding_schema.json'),
                'training_seed': conf['training_seed'], 'rounds': conf['rounds'],
                'condition': condition, 'configuration': mode,
                'isolated_interpreter': bool(sys.flags.isolated),
                'bytecode_writes_disabled': bool(sys.dont_write_bytecode),
                'pid': os.getpid()}
    if native:
        module = importlib.import_module(
            ('ehr_secagg' if mode == 'C' else 'ehr_fedavg') + '.bundle_checks')
        if not Path(module.__file__).resolve().is_relative_to(app):
            raise RuntimeError('Bundle checker imported outside selected fixture')
        from ehr_fedavg.host import local_environment
        original = module.probe_unpacked_fab

        def no_benchmark_data(fab, appdir, work, runtime, env):
            # The fixture contains no benchmark arrays. Native building, exact
            # contents/ledger checks, isolated model and callback imports remain
            # enabled. Actual PrepareRepair still probes full client inputs.
            return original(fab, appdir, work, None, env, require_flower=True)

        with patch.object(module, 'probe_unpacked_fab', side_effect=no_benchmark_data):
            audit = module.preflight_native_fab(
                app, job, condition, report, local_environment(job / 'probe_flower_home'))
    else:
        audit = {'status': 'PASS', 'source_resource_identity': verified,
                 'native_flower_build_executed': False,
                 'services_started_by_this_gate': False,
                 'training_or_validation_metrics_computed': False,
                 'test_data_opened': False}
    audit['test_fixture_import'] = identity
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', required=True, type=Path)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('B', 'C'))
    parser.add_argument('--condition', required=True, choices=('reference', 'noniid'))
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    rc = 0
    try:
        audit = _child(args.app.resolve(), args.job.resolve(), args.mode,
                       args.condition, args.report.resolve(), args.native)
    except Exception as exc:
        # Preserve a production native-failure report, when one was already
        # written, and surface the actual error to unittest without hiding it.
        audit = json.loads(args.report.read_text()) if args.report.is_file() else {}
        audit.update(status='FAIL', error_type=type(exc).__name__, message=str(exc))
        traceback.print_exc()
        rc = 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
