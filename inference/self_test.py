"""Functional self-check using only bundled synthetic CSV fixtures."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import csv
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'verification/self_test_latest.json')
    args = parser.parse_args()
    cases = []
    def record(name, detail):
        cases.append({'name': name, 'status': 'PASS', 'detail': detail})
    report = {'status': 'RUNNING', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'python': sys.version, 'cases': cases,
              'fixture': 'Bundled hand-constructed synthetic samples only', 'new_training': False}
    try:
        import numpy as np
        import torch
        from zoneinfo import ZoneInfo
        from demo_engine import InferenceEngine
        ZoneInfo('Asia/Kuala_Lumpur')
        report['packages'] = {'numpy': np.__version__, 'torch': torch.__version__}
        engine = InferenceEngine(ROOT / 'runtime_config.json')
        assert all(path.is_relative_to(ROOT) for path in engine.paths.values())
        assert all(path.exists() for path in engine.paths.values())
        record('all_paths_inside_extracted_bundle', True)
        record('frozen_asset_hashes_and_original_model_binding', engine.model_info)
        expected = json.loads((ROOT / 'samples/Expected_Results.json').read_text())
        for sample in expected['samples']:
            text = (ROOT / 'samples' / sample['file']).read_text(encoding='utf-8')
            if sample['expected'] == 'Rejected':
                try:
                    engine.score_csv(text, 'C1')
                except ValueError as exc:
                    assert str(exc) == sample['error'], (sample['file'], str(exc))
                    record(sample['file'], str(exc))
                else:
                    raise AssertionError(f"Invalid sample accepted: {sample['file']}")
            else:
                result = engine.score_csv(text, 'C1')
                assert math.isclose(result['score'], sample['score'], abs_tol=1e-6, rel_tol=1e-6)
                assert result['threshold'] == sample['threshold']
                assert result['alert'] == (sample['expected'] == 'Review recommended')
                assert all(result['integrity'].values())
                assert math.isclose(sum(event['score_contribution'] for event in result['events']),
                                    result['score'], abs_tol=2e-6, rel_tol=2e-6)
                record(sample['file'], {'score': result['score'], 'threshold': result['threshold'],
                                       'alert': result['alert']})
        text = (ROOT / 'samples/01_below_threshold.csv').read_text(encoding='utf-8')
        for client in ('C2', 'C3'):
            result = engine.score_csv(text, client)
            assert result['threshold'] == engine.calibration['clients'][client]['thresholds']['0.99']
            assert not result['alert']
            record('frozen_threshold_' + client, result['threshold'])
        assert not any(parameter.grad is not None for parameter in engine.model.parameters())
        record('no_training_or_gradients', True)
        report['status'] = 'PASS'
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error'] = f'{type(exc).__name__}: {exc}'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"{report['status']}: {len(cases)} portable inference checks")
    if report['status'] != 'PASS':
        print(report.get('error', 'Unknown failure'))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
