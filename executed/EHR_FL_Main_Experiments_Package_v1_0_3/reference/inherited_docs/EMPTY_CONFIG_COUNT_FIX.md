# 1.0.2 — Read-only ConfigRecord zero-length counting repair

## Failure evidence

The submitted run `20260908T122123Z_57a5576b` completed 307 of 308 test cases.
One native public-vector protocol test errored during `unmask_stage` at
`ConfigRecord.count_bytes(): v[0]` (`IndexError`). The seven read-only-message-ID
errors repaired in 1.0.1 are absent. There are no completed measured runs and the
report records that Deployment Engine did not start. The 17 manifest-listed
report files match their hashes/sizes; original and patched environment snapshots
are byte-identical before/after. Those snapshots are not a whole-machine audit.

The pinned source constructs `dead_nids=[]` when all clients are active. Empty
lists are valid protocol metadata, not missing EHR observations. The report does
not export the actual field values; the traceback proves empty-list indexing,
while `dead_nids` is the source-supported explanation of this unmask context.

## Minimal runtime change

Only `app/mai_measurement.py` changes in the runtime code. Its config-size helper
recognises zero-length lists/strings/bytes, counts the remaining entries in a
new temporary ConfigRecord using the native method, and adds the lengths of ALL
excluded keys. The actual transmitted ConfigRecord/Message is unchanged. No key
or list is deleted from the live payload and no dummy client ID is inserted.
A `dead_nids=[]` item contributes zero value-element bytes and nine logical key
units under this counting rule; it does NOT make the complete message zero bytes.

Non-empty configs keep native counting, including booleans, zero numeric values
and heterogeneous-length string/bytes lists. ArrayRecord/MetricRecord counts
remain native. Other exceptions and invalid native counter results still fail.
There is no try/except IndexError catch, no fallback to zero for a whole record,
and no monkeypatch or reinstall of Flower. Empty scalar text/bytes receive the
same explicitly documented zero-length treatment to avoid the related native
get_var_bytes error; this is not a change to any protocol values.

This is an application **logical record-content** count, not serialized protobuf,
gRPC/TCP/IP bytes, Python object memory size or cryptographic overhead alone.
The pinned native `len(str)`/`len(key)` convention is preserved, including for
non-ASCII strings; no claim of UTF-8 wire-byte accuracy is made. Counter overhead
remains within the previously disclosed instrumented-pipeline timing scope.

Rule ID `flower122_record_content_zero_length_v1` is included in client and server
measurement reports. Both ends use the same repair and are reconciled; totals
are still counted only once at the client boundary. No update values, masks,
shares or secrets are exported by this change.

## Unchanged study and execution

Same 3 paired repetitions, B-C / C-B / B-C; Reference; 6 rounds; round 1 warm-up;
256/128 routine train/validation sessions per client; same initialisation, seed,
LSTM, optimiser, 119-dimensional representation and HMAC PRG. Neither final test
partition is opened. All failed attempts are retained; no automatic trial retry.
The original metadata-fixture repair is retained and the source package stays
untouched. No old measurements exist to combine with this new measurement run.

## Test interpretation

The new suite has 363 cases: 308 inherited and 55 added. In preparation 307
completed successfully and 56 native Flower cases could not run because Flower
is unavailable. Local tests execute the verbatim upstream count METHOD inside a
clearly labelled dictionary harness, plus stand-in boundary tests; this is not
an installed Flower class or Deployment Engine. Native cases additionally check
serialized content identity, both boundary observers, empty status values, and
the public off-lattice SecAgg+ protocol. The target runner requires all 363 with
zero failures/errors/skips; the previously failing native test is retained.

The metadata repair logs in `reports_preparation/metadata_fix/` are historical
1.0.1 evidence, not results of this repair. New evidence is in
`reports_preparation/empty_config_fix/`. No native deployment or benchmark training
was executed in preparation. Privacy certification and main-study freeze remain
false independently of the outcome of these functional counting checks.
