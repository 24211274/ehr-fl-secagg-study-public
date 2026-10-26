"""Exercise the public demo with synthetic files and isolated visitor cookies.

The helper starts and stops only its own local Uvicorn subprocess. It requires
the application's installed runtime, but its HTTP client uses the standard
library only. It never submits the user's new file or historical benchmark.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import http.client
from http.cookiejar import CookieJar
from http.cookies import SimpleCookie
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener


MAX_BODY = 1024 * 1024
EXPECTED_CHECKPOINT = '79d631dd38f0e1e8701dcea02e7a3ddb86492652d2844e5b6f7fcbb883bc54fc'


class Visitor:
    def __init__(self, base):
        self.base = base
        self.cookies = CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookies))
        self.token = None

    def request(self, path, *, payload=None, raw=None, token=True, origin=True):
        headers = {}
        body = raw
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        if body is not None:
            headers['Content-Type'] = 'application/json'
        if token is True and self.token:
            headers['X-Demo-Token'] = self.token
        elif isinstance(token, str):
            headers['X-Demo-Token'] = token
        if origin is True:
            headers['Origin'] = self.base
        elif isinstance(origin, str):
            headers['Origin'] = origin
        request = Request(self.base + path, data=body, headers=headers)
        try:
            response = self.opener.open(request, timeout=20)
        except HTTPError as error:
            response = error
        with response:
            data = response.read()
            content_type = response.headers.get('Content-Type', '')
            value = json.loads(data.decode('utf-8')) if 'application/json' in content_type else data
            return response.status, value, dict(response.headers), response.headers.get_all('Set-Cookie', [])

    def configure(self):
        status, value, headers, cookie_headers = self.request('/api/config', token=False)
        if status != 200:
            raise AssertionError(f'Configuration failed with HTTP {status}: {value}')
        self.token = value['token']
        return value, headers, cookie_headers


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spawn(app_dir, port, *, secure=False, ttl='1200', rate='60'):
    with socket.socket() as probe:
        probe.settimeout(.3)
        if probe.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'QA port {port} is already in use; no existing process will be stopped')
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    environment.update({'COOKIE_SECURE': '1' if secure else '0',
                        'SESSION_TTL_SECONDS': ttl, 'SCORING_RATE_PER_MINUTE': rate,
                        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'})
    process = subprocess.Popen(
        [sys.executable, '-B', '-m', 'uvicorn', 'web_app:app', '--host', '127.0.0.1',
         '--port', str(port), '--workers', '1', '--no-access-log'],
        cwd=app_dir, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace',
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0,
    )
    base = f'http://127.0.0.1:{port}'
    client = Visitor(base)
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log, _ = process.communicate()
            raise RuntimeError(f'QA server exited before readiness: {log[-4000:]}')
        try:
            status, _, _, _ = client.request('/healthz', token=False)
            if status == 200:
                return process, base
        except (URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(.2)
    stop(process)
    raise RuntimeError('QA server did not become ready within 50 seconds')


def stop(process):
    if process is None:
        return ''
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log, _ = process.communicate()
    return log or ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'qa/Cloud_API_Verification.json')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    app_dir = args.app_dir.resolve(strict=True)
    evidence = {'status': 'RUNNING', 'scope': 'Actual isolated public API functional checks',
                'fixtures': 'Bundled synthetic CSV samples only',
                'user_new_file_processed': False, 'human_investigation_performed': False,
                'new_training': False, 'main_server_port_touched': False, 'cases': []}
    secrets = []
    process = None
    server_logs = []

    def record(name, detail=True):
        evidence['cases'].append({'name': name, 'status': 'PASS', 'detail': detail})

    def expect_response(client, path, expected_status, *, name, **kwargs):
        status, value, headers, cookies = client.request(path, **kwargs)
        assert status == expected_status, f'{name}: expected HTTP {expected_status}, got {status}: {value}'
        if expected_status >= 400:
            assert isinstance(value, dict) and isinstance(value.get('error'), str), (name, value)
        record(name, {'status': status, 'message': value.get('error') if isinstance(value, dict) else None})
        return value, headers, cookies

    try:
        configuration = json.loads((app_dir / 'runtime_config.json').read_text(encoding='utf-8'))
        immutable = {name: app_dir / relative for name, relative in configuration['paths'].items()
                     if name != 'source_app'}
        hashes = {name: sha(path) for name, path in immutable.items()}
        json_before = {str(path.relative_to(app_dir)) for path in app_dir.rglob('*.json')}
        csv_before = {str(path.relative_to(app_dir)) for path in app_dir.rglob('*.csv')}
        process, base = spawn(app_dir, args.port)
        a, b = Visitor(base), Visitor(base)
        expect_response(a, '/healthz', 200, name='health_readiness')
        page, _, _ = expect_response(a, '/', 200, name='English_page_available')
        assert b'<html lang="en">' in page and b'Session Check' in page
        config_a, headers_a, cookies_a = a.configure()
        config_b, _, cookies_b = b.configure()
        secrets.extend([a.token, b.token])
        assert a.token != b.token and len(a.token) >= 24 and len(b.token) >= 24
        assert config_a['clients'] == ['C1', 'C2', 'C3']
        assert config_a['model']['checkpoint_sha256'] == EXPECTED_CHECKPOINT
        assert config_a['model']['configuration'] == 'C'
        assert config_a['model']['condition'] == 'reference' and config_a['model']['seed'] == 11
        record('separate_visitor_tokens_and_original_model', {'tokens_differ': True,
                'checkpoint_sha256': config_a['model']['checkpoint_sha256']})
        cookie_a, cookie_b = SimpleCookie(), SimpleCookie()
        for header in cookies_a:
            cookie_a.load(header)
        for header in cookies_b:
            cookie_b.load(header)
        assert cookie_a and cookie_b
        values_a = {morsel.value for morsel in cookie_a.values()}
        values_b = {morsel.value for morsel in cookie_b.values()}
        secrets.extend(values_a | values_b)
        assert values_a.isdisjoint(values_b)
        assert all(morsel['httponly'] and morsel['samesite'].lower() == 'lax'
                   for morsel in cookie_a.values())
        record('visitor_cookie_is_HttpOnly_and_SameSite_Lax', True)
        assert 'no-store' in headers_a.get('cache-control', headers_a.get('Cache-Control', ''))
        csp = headers_a.get('content-security-policy', headers_a.get('Content-Security-Policy', ''))
        assert "default-src 'self'" in csp and "object-src 'none'" in csp
        assert headers_a.get('x-content-type-options', headers_a.get('X-Content-Type-Options')) == 'nosniff'
        record('no_store_and_security_response_headers', True)
        template, _, _ = expect_response(a, '/input-template.csv', 200, name='template_download')
        assert template == (app_dir / 'input_template.csv').read_bytes()
        below = (app_dir / 'samples/01_below_threshold.csv').read_text(encoding='utf-8')
        above = (app_dir / 'samples/02_above_threshold.csv').read_text(encoding='utf-8')
        expect_response(a, '/api/score', 403, payload={'csv': below, 'client': 'C1'},
                        token=False, name='missing_CSRF_token_rejected')
        expect_response(a, '/api/score', 403, payload={'csv': below, 'client': 'C1'},
                        token=b.token, name='other_visitors_CSRF_token_rejected')
        expect_response(a, '/api/score', 403, payload={'csv': below, 'client': 'C1'},
                        origin='https://foreign.example', name='foreign_origin_rejected')
        result, _, _ = expect_response(a, '/api/score', 200,
            payload={'csv': below, 'client': 'C1'}, name='C1_actual_below_threshold_inference')
        assert math.isclose(result['score'], .19823519885540009, abs_tol=1e-6, rel_tol=1e-6)
        assert result['threshold'] == .21659636572003307 and result['alert'] is False
        assert result['event_count'] == 12 and all(result['integrity'].values())
        assert abs(math.fsum(e['score_contribution'] for e in result['events']) - result['score']) < 2e-6
        evidence['cases'][-1]['detail'].update({'score': result['score'],
            'threshold': result['threshold'], 'alert': result['alert'], 'events': result['event_count']})
        run_id = result['run_id']
        note = 'QA ONLY public API review; no human investigation.'
        expect_response(b, '/api/review', 404, payload={'run_id': run_id, 'decision': 'needs_context',
                        'note': note}, name='another_visitor_cannot_review_A_result')
        expect_response(b, '/api/result/' + run_id, 404, name='no_public_result_read_endpoint')
        review, _, _ = expect_response(a, '/api/review', 200,
            payload={'run_id': run_id, 'decision': 'needs_context', 'note': note},
            name='own_review_acknowledged_in_memory')
        assert review['saved'] is True and review['status'] == 'recorded'
        assert review['run_id'] == run_id and review['note'] == note
        expect_response(a, '/api/review', 400,
            payload={'run_id': run_id, 'decision': 'invalid', 'note': note},
            name='invalid_review_decision_rejected')
        expect_response(a, '/api/review', 400,
            payload={'run_id': run_id, 'decision': 'needs_context', 'note': 'x' * 4001},
            name='oversized_review_note_rejected')
        expect_response(a, '/api/review', 400,
            payload={'run_id': run_id, 'decision': 'needs_context', 'note': ['not', 'text']},
            name='non_text_review_note_rejected')
        expect_response(a, '/api/review', 404,
            payload={'run_id': ['not', 'a', 'run_id'], 'decision': 'needs_context', 'note': note},
            name='non_text_run_id_rejected')
        expect_response(a, '/api/review', 404,
            payload={'run_id': 'not-a-current-run', 'decision': 'needs_context', 'note': note},
            name='unknown_run_review_rejected')
        for client, threshold in [('C2', .22253612235188477), ('C3', .20509986847639133)]:
            selected, _, _ = expect_response(a, '/api/score', 200,
                payload={'csv': below, 'client': client}, name=client + '_original_client_threshold')
            assert selected['threshold'] == threshold and selected['alert'] is False
            evidence['cases'][-1]['detail'].update({'threshold': selected['threshold'], 'alert': selected['alert']})
        high, _, _ = expect_response(a, '/api/score', 200,
            payload={'csv': above, 'client': 'C1'}, name='actual_above_threshold_inference')
        assert math.isclose(high['score'], 1.7077388763427734, abs_tol=1e-6, rel_tol=1e-6)
        assert high['alert'] is True and high['threshold'] == .21659636572003307
        evidence['cases'][-1]['detail'].update({'score': high['score'],
            'threshold': high['threshold'], 'alert': high['alert'], 'events': high['event_count']})
        expect_response(a, '/api/review', 404,
            payload={'run_id': run_id, 'decision': 'needs_context', 'note': note},
            name='oldest_run_is_evicted_after_three_newer_results')
        fixture_row = next(csv.DictReader(io.StringIO(below)))
        first_time = datetime.fromisoformat(fixture_row['timestamp_utc'])
        max_buffer = io.StringIO(newline='')
        max_writer = csv.DictWriter(max_buffer, fieldnames=list(fixture_row))
        max_writer.writeheader()
        for position in range(512):
            max_writer.writerow({**fixture_row, 'session_id': 'QA-CLOUD-MAX-512',
                'event_id': f'QA-CLOUD-MAX-E{position:03d}', 'sequence_position': str(position),
                'timestamp_utc': (first_time + timedelta(seconds=45 * position)).isoformat()})
        started = time.perf_counter()
        maximum, _, _ = expect_response(a, '/api/score', 200,
            payload={'csv': max_buffer.getvalue(), 'client': 'C1'}, name='maximum_512_event_session_processed')
        assert maximum['event_count'] == 512 and len(maximum['events']) == 512
        assert math.isfinite(maximum['score']) and all(maximum['integrity'].values())
        evidence['cases'][-1]['detail'].update({'event_count': maximum['event_count'],
            'elapsed_seconds': time.perf_counter() - started,
            'score_is_finite': True, 'synthetic_fixture_without_threat_label': True})
        expect_response(a, '/api/score', 400, payload={'csv': 'bad,column\ninvalid,row\n', 'client': 'C1'},
                        name='malformed_CSV_rejected')
        expect_response(a, '/api/score', 400, payload={'csv': below, 'client': 'C4'},
                        name='unknown_client_rejected')
        expect_response(a, '/api/score', 400, payload={'client': 'C1'}, name='missing_CSV_rejected')
        expect_response(a, '/api/score', 400, payload=['not', 'an', 'object'],
                        name='non_object_JSON_rejected')
        expect_response(a, '/api/score', 400, raw=b'{"csv":', name='invalid_JSON_text_rejected')
        expect_response(a, '/api/score', 413,
                        payload={'csv': 'x' * (512 * 1024 + 1), 'client': 'C1'},
                        name='CSV_byte_limit_enforced_within_request_limit')
        expect_response(a, '/api/score', 400,
                        payload={'csv': 'x' * 200000 + '\nanything\n', 'client': 'C1'},
                        name='oversized_CSV_field_rejected_without_server_error')
        # The server rejects an oversized Content-Length before consuming its
        # body. Sending a megabyte through urllib can therefore race a valid
        # early response and raise BrokenPipe in the client. Exercise that
        # early rejection directly without writing a body after the rejection.
        connection = http.client.HTTPConnection('127.0.0.1', args.port, timeout=20)
        connection.putrequest('POST', '/api/score')
        connection.putheader('Content-Type', 'application/json')
        connection.putheader('Content-Length', str(MAX_BODY + 1))
        connection.putheader('Origin', a.base)
        connection.putheader('X-Demo-Token', a.token)
        connection.putheader('Cookie', '; '.join(f'{cookie.name}={cookie.value}' for cookie in a.cookies))
        connection.endheaders()
        oversized_response = connection.getresponse()
        oversized_error = json.loads(oversized_response.read())
        assert oversized_response.status == 413 and isinstance(oversized_error.get('error'), str)
        connection.close()
        record('request_size_limit_enforced', {'status': 413, 'message': oversized_error['error']})
        for path in ('/assets/selected_state.pt', '/runtime_config.json', '/source_app/pilot_core/model.py',
                     '/samples/Expected_Results.json', '/Dockerfile', '/.git/config', '/requirements_web.txt'):
            expect_response(a, path, 404, name='private_file_not_served_' + path.rsplit('/', 1)[-1])
        assert {name: sha(path) for name, path in immutable.items()} == hashes
        record('all_frozen_assets_unchanged', list(hashes))
        assert not (app_dir / 'runs').exists(), 'Public service must not create a runs directory'
        assert not list(app_dir.rglob('human_reviews.jsonl'))
        assert {str(path.relative_to(app_dir)) for path in app_dir.rglob('*.json')} == json_before
        assert {str(path.relative_to(app_dir)) for path in app_dir.rglob('*.csv')} == csv_before
        record('no_uploaded_files_results_or_reviews_written_to_disk', True)
        if sys.platform.startswith('linux'):
            status_file = Path(f'/proc/{process.pid}/status').read_text()
            memory_kib = {line.split(':', 1)[0]: int(line.split()[1])
                          for line in status_file.splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))}
            evidence['resident_memory_mib'] = {key: value / 1024 for key, value in memory_kib.items()}
        server_logs.append(stop(process))
        process = None
        # Verify the production default cookie independently. Secure cookies are
        # intentionally not returned by CookieJar to this HTTP-only QA server.
        process, secure_base = spawn(app_dir, args.port, secure=True)
        secure_visitor = Visitor(secure_base)
        _, _, secure_cookie_headers = secure_visitor.configure()
        production_cookie = SimpleCookie()
        for header in secure_cookie_headers:
            production_cookie.load(header)
        assert production_cookie and all(m['secure'] and m['httponly']
                                         and m['samesite'].lower() == 'lax'
                                         for m in production_cookie.values())
        secrets.extend(m.value for m in production_cookie.values())
        record('production_cookie_is_Secure_HttpOnly_SameSite_Lax', True)
        server_logs.append(stop(process))
        process = None
        # Short TTL is a test-only configuration. Expired browser tokens must
        # fail, then a new configuration request must create a fresh session.
        process, expiry_base = spawn(app_dir, args.port, secure=False, ttl='1')
        expiry_visitor = Visitor(expiry_base)
        expiry_visitor.configure()
        expired_token = expiry_visitor.token
        secrets.append(expired_token)
        time.sleep(1.3)
        expect_response(expiry_visitor, '/api/score', 403,
                        payload={'csv': below, 'client': 'C1'}, name='expired_session_token_rejected')
        renewed, _, _ = expiry_visitor.configure()
        secrets.append(renewed['token'])
        assert renewed['token'] != expired_token
        record('configuration_renews_expired_session', {'new_token_differs': True})
        server_logs.append(stop(process))
        process = None
        # A deterministic one-score budget exercises the rate gate without
        # relying on request timing, thread races or a modified scoring model.
        process, rate_base = spawn(app_dir, args.port, secure=False, rate='1')
        rate_visitor = Visitor(rate_base)
        rate_visitor.configure()
        secrets.append(rate_visitor.token)
        limited_first, _, _ = expect_response(rate_visitor, '/api/score', 200,
            payload={'csv': below, 'client': 'C1'}, name='rate_budget_allows_first_inference')
        assert math.isclose(limited_first['score'], .19823519885540009, abs_tol=1e-6, rel_tol=1e-6)
        expect_response(rate_visitor, '/api/score', 429,
                        payload={'csv': below, 'client': 'C1'}, name='per_visitor_rate_budget_rejects_second_inference')
        rate_token = rate_visitor.token
        rate_visitor.configure()
        assert rate_visitor.token == rate_token
        expect_response(rate_visitor, '/api/score', 429,
                        payload={'csv': below, 'client': 'C1'}, name='configuration_refresh_does_not_reset_rate_budget')
        other_rate_visitor = Visitor(rate_base)
        other_rate_visitor.configure()
        secrets.append(other_rate_visitor.token)
        expect_response(other_rate_visitor, '/api/score', 200,
                        payload={'csv': below, 'client': 'C1'}, name='separate_visitor_retains_own_rate_budget')
        evidence['status'] = 'PASS'
    except Exception as error:
        evidence['status'] = 'FAIL'
        evidence['failure'] = f'{type(error).__name__}: {error}'
    finally:
        server_logs.append(stop(process))
        combined_log = '\n'.join(server_logs)
        for secret in secrets:
            combined_log = combined_log.replace(secret, '[REDACTED_SESSION_TOKEN]')
        evidence['server_log'] = combined_log[-6000:]
        evidence['case_count'] = len(evidence['cases'])
        evidence['isolated_servers_stopped'] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status': evidence['status'], 'case_count': evidence['case_count'],
                      'output': str(args.output), 'failure': evidence.get('failure')}, indent=2))
    return 0 if evidence['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
