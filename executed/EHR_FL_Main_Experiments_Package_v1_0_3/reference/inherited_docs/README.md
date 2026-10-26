# EHR FL timing/payload pilot 1.0.2

Run the predeclared three paired Reference timing repetitions in the accepted
patched Flower environment. Six service runs in B-C, C-B, B-C order; six rounds
per run, first round warm-up. Original 256/128 routine sessions per client, not
full-size or final-test data. No installation or model/PRG modification.

Read `TIMING_ACCEPTANCE.md` for frozen workload, instrumentation boundaries,
analysis, failure rules and claim limits. See `README_AR.md` for exact Windows/WSL
commands and reporting. The primary timer is the instrumented server fit workflow,
not pure cryptography or uninstrumented production time. Application record-content
bytes are checked at both ends and counted once at the clients; full network bytes
are NOT measured. Three paired run observations, not thirty independent rounds.

The original 1.0.0 timing implementation adds read-only counters/timers around accepted numerical paths and
changes the workload to the previously planned TIMING rows. `fit_one_client`, model,
loss, optimiser, schema, preprocessing coefficients and PRG source are unchanged.
`reference/unchanged_numerical_source.json` records checkable source identities.

Preparation tests without Flower use explicitly labelled stand-ins. The target
runner requires the complete native test suite, all six FAB load probes, prior
accepted report checks, all rounds and exact two-boundary count reconciliation.
No skipped runtime test is a pass. No automatic training retries. All failed and
warm-up observations are retained; no comparative success summary on a partial run.

Inference microbenchmarks, extra-seed sanity, final model tuning, wire traffic
measurement and main-study freeze are not performed by this package.

## Historical 1.0.1: test-fixture compatibility repair only

Four assignments across two test modules wrongly treated Flower 1.22
`Metadata.message_id` as writable; the user's 285-test run recorded seven errors
at those sites (278 successful cases, zero assertion failures, zero skips).
The new `tests/flower_message_fixture.py` constructs received test instructions
using `Metadata(...)` and the pinned internal `Message(metadata=..., content=...)`
constructor. It deep-copies content, preserves routing/run/TTL/correlation and
existing request IDs, and never mutates the source Message or the framework.
It is test-only and is not copied into FAB apps. Production measurement still
requires real SuperLink-assigned IDs and rejects duplicate or missing IDs.

In 1.0.1, all runtime application, model, security, environment, measurement and
analysis code, configurations/manifests, and `TIMING_ACCEPTANCE.md` were
byte-identical to 1.0.0. No numerical or timing policy has been loosened to obtain a pass.
The 1.0.1 target suite contained 308 cases: original 285 plus 23 fixture regressions.
Here 281 completed successfully; 27 native Flower cases could not run because
Flower is not installed. The target host must execute all cases without skips.
No native Flower deployment or measured training was executed in preparation.
See `METADATA_FIX.md`; 1.0.0 preparation reports are retained as historical evidence.

## Current 1.0.2: zero-length config content counting

See `EMPTY_CONFIG_COUNT_FIX.md`. This changes the application observer, not only
its tests. Empty lists/text/bytes contribute zero value bytes but retain their
key lengths; non-empty values and other record types retain native counts. The
live message and installed library are not modified. Rule ID:
`flower122_record_content_zero_length_v1`. Empty scalar status values are covered
as a related edge case. Scientific workload/configs/analysis remain unchanged.

The target suite has 363 cases (55 added). In preparation 307 passed; 56 native
Flower cases were not executable here. The target must execute all with no skips.
No native deployment or timing run was performed in preparation. Source-method
harness checks are not presented as native Flower protocol execution.
