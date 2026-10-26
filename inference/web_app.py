"""Public, session-isolated inference. Uploaded data and reviews stay in RAM only."""
from __future__ import annotations

import asyncio
import csv
from collections import OrderedDict, deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from demo_engine import InferenceEngine

ROOT = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 1024 * 1024
MAX_CSV_BYTES = 512 * 1024
SESSION_TTL = max(1, int(os.environ.get('SESSION_TTL_SECONDS', '1200')))
MAX_SESSIONS = max(1, min(128, int(os.environ.get('MAX_SESSIONS', '128'))))
MAX_RESULTS_PER_SESSION = 3
MAX_RESULT_CACHE_BYTES = 16 * 1024 * 1024
SCORE_RATE = max(1, int(os.environ.get('SCORING_RATE_PER_MINUTE', '12')))
GLOBAL_SCORE_RATE = max(1, int(os.environ.get('GLOBAL_SCORING_RATE_PER_MINUTE', '120')))
COOKIE_SECURE = os.environ.get('COOKIE_SECURE', '1').lower() not in ('0', 'false', 'no')
COOKIE_NAME = 'session_check_session'
PUBLIC_ORIGIN = os.environ.get('PUBLIC_ORIGIN', '').rstrip('/')
EXTERNAL_HOST = os.environ.get('RENDER_EXTERNAL_HOSTNAME', '').strip().lower()
if PUBLIC_ORIGIN:
    parsed_origin = urlsplit(PUBLIC_ORIGIN)
    if parsed_origin.scheme not in ('https', 'http') or not parsed_origin.hostname or parsed_origin.path:
        raise RuntimeError('PUBLIC_ORIGIN must be an absolute HTTP(S) origin without a path')
else:
    parsed_origin = None
ALLOWED_HOSTS = sorted({'127.0.0.1', 'localhost'} | ({EXTERNAL_HOST} if EXTERNAL_HOST else set())
                       | ({parsed_origin.hostname} if parsed_origin else set()))


@dataclass
class Session:
    csrf: str
    expires: float
    results: OrderedDict = field(default_factory=OrderedDict)
    scores: deque = field(default_factory=deque)


class SessionStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self.result_order: OrderedDict[tuple[str, str], int] = OrderedDict()
        self.cached_bytes = 0
        self.global_scores = deque()

    def _drop_result(self, sid, run_id):
        size = self.result_order.pop((sid, run_id), 0)
        self.cached_bytes -= size
        session = self.sessions.get(sid)
        if session:
            session.results.pop(run_id, None)

    def _drop_session(self, sid):
        session = self.sessions.get(sid)
        if session:
            for run_id in list(session.results):
                self._drop_result(sid, run_id)
        self.sessions.pop(sid, None)

    def purge(self):
        with self.lock:
            now = time.monotonic()
            for sid in [sid for sid, value in self.sessions.items() if value.expires <= now]:
                self._drop_session(sid)

    def existing(self, sid):
        with self.lock:
            self.purge()
            session = self.sessions.get(sid)
            if session is not None:
                session.expires = time.monotonic() + SESSION_TTL
                self.sessions.move_to_end(sid)
            return session

    def get_or_create(self, sid):
        with self.lock:
            session = self.existing(sid)
            if session is not None:
                return sid, session
            if len(self.sessions) >= MAX_SESSIONS:
                raise HTTPException(503, 'The service is busy. Please try again shortly.')
            sid = secrets.token_urlsafe(32)
            session = Session(secrets.token_urlsafe(32), time.monotonic() + SESSION_TTL)
            self.sessions[sid] = session
            return sid, session

    def allow_score(self, session):
        with self.lock:
            now = time.monotonic()
            for values in (session.scores, self.global_scores):
                while values and values[0] <= now - 60:
                    values.popleft()
            if len(session.scores) >= SCORE_RATE or len(self.global_scores) >= GLOBAL_SCORE_RATE:
                raise HTTPException(429, 'Too many inference requests. Please wait one minute.')
            session.scores.append(now)
            self.global_scores.append(now)

    def store_result(self, sid, session, result):
        with self.lock:
            # A session could expire during a slow inference, but no visitor can inherit it.
            if self.sessions.get(sid) is not session:
                raise HTTPException(403, 'Your session expired. Select Reconnect and run inference again.')
            run_id = result['run_id']
            size = len(json.dumps(result, ensure_ascii=False).encode('utf-8'))
            if size > MAX_RESULT_CACHE_BYTES:
                raise HTTPException(413, 'The result is too large for this demonstration.')
            while len(session.results) >= MAX_RESULTS_PER_SESSION:
                self._drop_result(sid, next(iter(session.results)))
            while self.result_order and self.cached_bytes + size > MAX_RESULT_CACHE_BYTES:
                self._drop_result(*next(iter(self.result_order)))
            session.results[run_id] = result
            self.result_order[(sid, run_id)] = size
            self.cached_bytes += size

    def review(self, sid, session, run_id, decision, note):
        with self.lock:
            result = session.results.get(run_id)
            if result is None:
                raise HTTPException(404, 'Result not found in your session. Run inference again.')
            review = {'run_id': run_id, 'decision': decision, 'note': note,
                      'recorded_utc': datetime.now(timezone.utc).isoformat(),
                      'reviewer_identity': 'current_browser_session_unverified',
                      'scope': 'User-entered demonstration review; not an independently verified investigation'}
            self._drop_result(sid, run_id)
            result['human_review'] = {'status': 'recorded', **review}
            self.store_result(sid, session, result)
            return {'saved': True, 'status': 'recorded', **review}


store = SessionStore()
model_lock = threading.Lock()


async def cleanup_sessions():
    while True:
        await asyncio.sleep(min(30, SESSION_TTL))
        store.purge()


@asynccontextmanager
async def lifespan(application):
    application.state.engine = await run_in_threadpool(InferenceEngine, ROOT / 'runtime_config.json')
    task = asyncio.create_task(cleanup_sessions())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        with store.lock:
            store.sessions.clear()
            store.result_order.clear()
            store.cached_bytes = 0


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


@app.middleware('http')
async def response_headers(request, call_next):
    try:
        response = await call_next(request)
    except Exception as error:
        # Log only the error class; never log CSVs, review notes, cookies or tokens.
        logging.getLogger('session_check').error('request_failed:%s', type(error).__name__)
        response = JSONResponse({'error': 'The request could not be processed. Please try again.'}, status_code=500)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_error(request, error):
    return JSONResponse({'error': str(error.detail)}, status_code=error.status_code)


def check_origin(request):
    if request.headers.get('sec-fetch-site') == 'cross-site':
        raise HTTPException(403, 'Use this service from its own page.')
    origin = request.headers.get('origin')
    if origin is None:
        return
    allowed = set()
    if PUBLIC_ORIGIN:
        allowed.add(PUBLIC_ORIGIN)
    if EXTERNAL_HOST:
        allowed.add('https://' + EXTERNAL_HOST)
    # Only local hostnames are accepted for the HTTP test/development origin.
    if request.url.hostname in ('localhost', '127.0.0.1'):
        allowed.add(str(request.base_url).rstrip('/'))
    if origin.rstrip('/') not in allowed:
        raise HTTPException(403, 'Use this service from its own page.')


def session_response(value, sid, status=200):
    response = JSONResponse(value, status_code=status)
    response.set_cookie(COOKIE_NAME, sid, max_age=SESSION_TTL, path='/',
                        secure=COOKIE_SECURE, httponly=True, samesite='lax')
    return response


def require_session(request):
    check_origin(request)
    sid = request.cookies.get(COOKIE_NAME)
    session = store.existing(sid)
    token = request.headers.get('x-demo-token', '')
    if session is None or not token or not secrets.compare_digest(session.csrf.encode('ascii'), token.encode('utf-8')):
        raise HTTPException(403, 'Your session expired. Select Reconnect and try again.')
    return sid, session


async def json_body(request):
    if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'application/json':
        raise HTTPException(400, 'A JSON request is required.')
    try:
        length = int(request.headers.get('content-length', '0'))
    except ValueError:
        raise HTTPException(400, 'Invalid request length.')
    if length < 0:
        raise HTTPException(400, 'Invalid request length.')
    if length > MAX_REQUEST_BYTES:
        raise HTTPException(413, 'Request is too large. Use one CSV session up to 512 KB.')
    chunks, size = [], 0
    try:
        async with asyncio.timeout(15):
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_REQUEST_BYTES:
                    raise HTTPException(413, 'Request is too large. Use one CSV session up to 512 KB.')
                chunks.append(chunk)
    except TimeoutError:
        raise HTTPException(408, 'The upload timed out. Please try again.')
    try:
        value = json.loads(b''.join(chunks))
    except (ValueError, UnicodeError):
        raise HTTPException(400, 'The request contains invalid JSON text.')
    if not isinstance(value, dict):
        raise HTTPException(400, 'A JSON object is required.')
    return value


@app.get('/', response_class=HTMLResponse)
async def page():
    return HTMLResponse((ROOT / 'index.html').read_text(encoding='utf-8'))


@app.get('/input-template.csv')
async def template():
    return Response((ROOT / 'input_template.csv').read_bytes(), media_type='text/csv; charset=utf-8')


@app.get('/healthz')
async def health():
    store.purge()
    return {'status': 'ok'}


@app.get('/api/config')
async def configuration(request: Request):
    check_origin(request)
    sid, session = store.get_or_create(request.cookies.get(COOKIE_NAME))
    return session_response({'model': request.app.state.engine.model_info,
        'clients': ['C1', 'C2', 'C3'], 'token': session.csrf,
        'scope': 'Temporary browser session; uploaded data and review notes are not saved to disk.'}, sid)


@app.post('/api/score')
async def score(request: Request):
    sid, session = require_session(request)
    data = await json_body(request)
    text, client = data.get('csv'), data.get('client', 'C1')
    if not isinstance(text, str):
        raise HTTPException(400, 'CSV text is required.')
    if not isinstance(client, str) or client not in ('C1', 'C2', 'C3'):
        raise HTTPException(400, 'Choose C1, C2 or C3.')
    if len(text.encode('utf-8')) > MAX_CSV_BYTES:
        raise HTTPException(413, 'CSV is too large. Use one session up to 512 KB.')
    if not model_lock.acquire(blocking=False):
        raise HTTPException(429, 'The model is busy. Please try again shortly.')
    try:
        store.allow_score(session)
        try:
            result = await run_in_threadpool(request.app.state.engine.score_csv, text, client)
        except (ValueError, TypeError, KeyError, csv.Error) as error:
            raise HTTPException(400, str(error))
        result['recorded_utc'] = datetime.now(timezone.utc).isoformat()
        result['storage'] = 'Temporary current-browser session only; not saved to disk.'
        store.store_result(sid, session, result)
        return session_response(result, sid)
    finally:
        model_lock.release()


@app.post('/api/review')
async def review(request: Request):
    sid, session = require_session(request)
    data = await json_body(request)
    run_id, decision, note = data.get('run_id'), data.get('decision'), data.get('note', '')
    if not isinstance(run_id, str) or len(run_id) > 80:
        raise HTTPException(404, 'Result not found in your session. Run inference again.')
    if decision not in ('needs_context', 'legitimate', 'escalate'):
        raise HTTPException(400, 'Choose a valid review decision.')
    if not isinstance(note, str) or len(note) > 4000:
        raise HTTPException(400, 'Review note must contain at most 4000 characters.')
    value = store.review(sid, session, run_id, decision, note)
    return session_response(value, sid)
