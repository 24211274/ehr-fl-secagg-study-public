"""New raw-session adapter; original preprocessing and learned model remain frozen."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alert_decision(score: float, threshold: float) -> bool:
    if not math.isfinite(score) or not math.isfinite(threshold):
        raise ValueError('Score and threshold must be finite')
    return score > threshold


class InferenceEngine:
    def __init__(self, config_path: Path):
        config_root = Path(config_path).resolve().parent
        self.config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        self.paths = {}
        for key, value in self.config['paths'].items():
            relative = Path(value)
            if relative.is_absolute() or ':' in value or value.startswith(('\\', '/')):
                raise ValueError('Portable configuration requires relative asset paths')
            target = (config_root / relative).resolve()
            if not target.is_relative_to(config_root):
                raise ValueError('Portable asset path escapes its package')
            self.paths[key] = target
        self.verify_frozen()
        sys.path.insert(0, str(self.paths['source_app']))
        import numpy as np
        import torch
        from torch.nn import functional as F
        from ehr_preprocessing import pipeline
        from ehr_preprocessing.dataset import collate_sessions
        from pilot_core.model import CompactLSTMAutoencoder, session_losses, state_digest
        self.np, self.torch, self.F = np, torch, F
        self.pipeline, self.collate = pipeline, collate_sessions
        self.session_losses, self.state_digest = session_losses, state_digest
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.schema = pipeline.approved_schema(self.paths['schema'])
        self.scaler = json.loads(self.paths['scaler'].read_text())
        self.calibration = json.loads(self.paths['calibration'].read_text())
        self.model_info = dict(self.config['model'])
        manifest = json.loads(self.paths['manifest'].read_text())
        rows = [r for r in manifest['models'] if all(r[k] == self.model_info[k]
                for k in ('configuration', 'condition', 'seed'))]
        if len(rows) != 1:
            raise ValueError('Selected-model manifest identity mismatch')
        selected = rows[0]
        digest = sha256(self.paths['checkpoint'])
        if (selected['selected_checkpoint_sha256'] != digest
                or self.calibration['selected_checkpoint_sha256'] != digest
                or selected['calibration_sha256'] != sha256(self.paths['calibration'])):
            raise ValueError('Checkpoint/calibration/manifest binding mismatch')
        if self.calibration['primary_quantile'] != .99:
            raise ValueError('Expected original primary q=.99')
        for k in ('configuration', 'condition'):
            if self.calibration[k] != self.model_info[k]:
                raise ValueError('Calibration model mismatch')
        if self.calibration['training_seed'] != self.model_info['seed']:
            raise ValueError('Calibration seed mismatch')
        self.model = CompactLSTMAutoencoder([len(f['categories'])
                                            for f in self.schema['categorical_fields']])
        self.model.load_state_dict(torch.load(self.paths['checkpoint'], map_location='cpu',
                                             weights_only=True), strict=True)
        self.model.eval()
        self.initial_digest = state_digest(self.model)
        self.model_info['checkpoint_sha256'] = digest
        self.fieldnames = ['session_id', 'event_id', 'sequence_position', 'timestamp_utc'] + [
            f['name'] for f in self.schema['categorical_fields']] + self.schema['binary_fields']

    def verify_frozen(self):
        for key, expected in self.config['sha256'].items():
            if sha256(self.paths[key]) != expected:
                raise ValueError(f'Frozen artefact changed: {key}')
        for relative, expected in self.config['source_sha256'].items():
            if sha256(self.paths['source_app'] / relative) != expected:
                raise ValueError(f'Original source changed: {relative}')

    def encode_rows(self, rows: list[dict], client: str):
        np, torch, p = self.np, self.torch, self.pipeline
        if client not in p.CLIENTS:
            raise ValueError('Choose C1, C2 or C3; calibration for a new institution is unavailable')
        if not 1 <= len(rows) <= 512:
            raise ValueError('A complete session must contain 1 to 512 events for this demo')
        if any(not isinstance(r, dict) or any(k not in r or not isinstance(r[k], str)
                                            for k in self.fieldnames) for r in rows):
            raise ValueError('Required CSV cells are missing')
        ids = {r['session_id'] for r in rows}
        if len(ids) != 1 or not next(iter(ids)).strip():
            raise ValueError('Submit exactly one nonempty session_id')
        if any(not r['event_id'].strip() for r in rows) or len({r['event_id'] for r in rows}) != len(rows):
            raise ValueError('event_id values must be nonempty and unique')
        if any('client' in r and r['client'] != client for r in rows):
            raise ValueError('CSV client disagrees with the selected client')
        if any('hospital_id' in r and r['hospital_id'] != p.CLIENTS[client] for r in rows):
            raise ValueError('CSV hospital_id disagrees with the selected client')
        if any(not re.fullmatch(r'0|[1-9][0-9]*', r['sequence_position']) for r in rows):
            raise ValueError('sequence_position must use canonical zero-based integers')
        rows = sorted(rows, key=lambda r: int(r['sequence_position']))
        if [int(r['sequence_position']) for r in rows] != list(range(len(rows))):
            raise ValueError('sequence_position must be unique, contiguous and start at zero')
        times = [p.parse_timestamp(r['timestamp_utc']) for r in rows]
        n = len(rows)
        categories = np.empty((n, 13), dtype='<i8')
        valid = np.empty((n, 13), dtype=bool)
        binary = np.empty((n, 3), dtype='<f4')
        unknown, missing = 0, 0
        for j, field in enumerate(self.schema['categorical_fields']):
            mapping = {v: k for k, v in enumerate(field['categories'])}
            for i, row in enumerate(rows):
                categories[i, j], valid[i, j] = p.encode_category(row[field['name']], field, mapping)
                unknown += int(categories[i, j] == field['unknown_index'])
                missing += int(categories[i, j] == field['missing_index'])
        for i, row in enumerate(rows):
            for j, name in enumerate(self.schema['binary_fields']):
                binary[i, j] = p.parse_boolean(row[name])
        timestamps, raw_time = p.temporal_values(times, self.schema[
            'preprocessing_policy_reference']['calendar_timezone'])
        sid = next(iter(ids))
        session = p.Session(sid, client, p.CLIENTS[client], 'new_inference',
                            [r['event_id'] for r in rows], timestamps, categories, valid,
                            binary, raw_time)
        x = torch.from_numpy(p.event_features(session, self.schema, self.scaler))
        sample = {'x': x, 'categorical': torch.from_numpy(categories),
                  'categorical_valid': torch.from_numpy(valid), 'binary': x[:, 110:113].clone(),
                  'temporal': x[:, 113:119].clone(), 'session_id': sid, 'client': client,
                  'partition': 'new_inference'}
        return sample, {'rows': rows, 'times': [t.isoformat() for t in times],
                        'coverage': {'valid_categorical_targets': int(valid.sum()),
                                     'total_categorical_targets': n * 13,
                                     'unknown_targets': unknown, 'missing_targets': missing}}

    def score_csv(self, text: str, client: str = 'C1') -> dict:
        self.verify_frozen()
        if not isinstance(text, str) or len(text.encode('utf-8')) > 524288:
            raise ValueError('CSV input must be at most 512 KiB')
        reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff'), newline=''))
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or not set(self.fieldnames).issubset(header):
            raise ValueError('Missing or duplicate CSV column names; use the supplied template')
        rows = []
        for row in reader:
            if None in row:
                raise ValueError('CSV row has more cells than the header')
            rows.append(row)
            if len(rows) > 512:
                raise ValueError('Demo limit is 512 events per complete session')
        t0 = time.perf_counter_ns()
        sample, metadata = self.encode_rows(rows, client)
        batch = self.collate([sample])
        t1 = time.perf_counter_ns()
        before = self.state_digest(self.model)
        if before != self.initial_digest:
            raise ValueError('In-memory model state changed')
        torch, F = self.torch, self.F
        with torch.no_grad():
            outputs = self.model(batch['x'], batch['lengths'])
            losses, event_losses, _ = self.session_losses(outputs, batch['categorical'],
                batch['binary'], batch['temporal'], batch['lengths'], batch['categorical_valid'])
        t2 = time.perf_counter_ns()
        after = self.state_digest(self.model)
        if after != before or any(v.grad is not None for v in self.model.parameters()):
            raise ValueError('Inference modified weights or created gradients')
        self.verify_frozen()
        score, n = float(losses[0]), len(rows)
        if not math.isfinite(score):
            raise ValueError('Nonfinite score')
        c = self.calibration['clients'][client]
        threshold = float(c['thresholds']['0.99'])
        if c['alert_operator'] != '>' or c['scale'] <= 0:
            raise ValueError('Unsupported calibration policy')
        events = []
        for i, row in enumerate(metadata['rows']):
            event_loss = float(event_losses[0, i])
            contribution = event_loss / n
            events.append({'event_id': row['event_id'], 'position': i,
                'timestamp_utc': metadata['times'][i], 'action': row['action'],
                'department': row['department'], 'resource_type': row['resource_type'],
                'event_loss': event_loss, 'score_contribution': contribution,
                'contribution_percent': 100 * contribution / score if score else 0.0})
        # Describe the adopted arithmetic without removing inputs or perturbing the model.
        cats, binary, temporal = outputs
        valid = batch['categorical_valid']
        cat_n = valid.sum(-1)
        cat_sum = torch.zeros_like(event_losses)
        for j, logits in enumerate(cats):
            target = batch['categorical'][:, :, j].clone().masked_fill(~valid[:, :, j], -100)
            cat_sum += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                                      reduction='none', ignore_index=-100).reshape(1, n)
        groups = 2 + (cat_n > 0).to(binary.dtype)
        family = {
            'categorical': float((cat_sum / cat_n.clamp_min(1) / groups).mean()),
            'boolean': float((F.binary_cross_entropy_with_logits(binary, batch['binary'],
                                          reduction='none').mean(-1) / groups).mean()),
            'temporal': float((F.smooth_l1_loss(temporal, batch['temporal'],
                                 reduction='none', beta=1.0).mean(-1) / groups).mean())}
        if not math.isclose(sum(e['score_contribution'] for e in events), score, abs_tol=2e-6, rel_tol=2e-6):
            raise ValueError('Event decomposition disagrees with original loss')
        if not math.isclose(sum(family.values()), score, abs_tol=2e-6, rel_tol=2e-6):
            raise ValueError('Family decomposition disagrees with original loss')
        alert = alert_decision(score, threshold)
        return {'run_id': str(uuid.uuid4()), 'session_id': sample['session_id'], 'client': client,
                'event_count': n, 'window': {'start': metadata['times'][0], 'end': metadata['times'][-1]},
                'score': score, 'standardised_score': (score - c['median']) / c['scale'],
                'threshold': threshold, 'quantile': .99, 'decision_operator': '>', 'alert': alert,
                'events': events, 'family_contributions': family, 'coverage': metadata['coverage'],
                'model': self.model_info, 'integrity': {'weights_unchanged': True,
                    'gradients_absent': True, 'frozen_artifacts_unchanged': True},
                'human_review': {'status': 'pending' if alert else 'not_triggered'},
                'timing_ms': {'preprocessing': (t1 - t0) / 1e6, 'model_and_score': (t2 - t1) / 1e6},
                'input_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                'notes': ['Complete-session inference; no partial-session threshold decisions.',
                          'Reconstruction score and standardised score are not threat probabilities.',
                          'Contributions describe reconstruction loss, not confirmed harmful events or causal importance.',
                          'New user inputs are not independently verified as unseen; the supplied example is hand-constructed synthetic data.',
                          'SecAgg+ was used during training and is not executed during inference.',
                          'Demo timings include local adapter checks and are not the thesis latency benchmark.']}
