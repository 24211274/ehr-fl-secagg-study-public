# Main-study measurement-scope correction — version 1.0.3

**Repair ID:** MAIN-MEASUREMENT-SCOPE-003. This is a code-defect correction, not a
transient infrastructure retry, a tuning decision, or a new research condition.

## Evidence and cause

The researcher supplied `server_result.json` and the traceback from B in campaign
`20260909T071824Z_83a874db`. They identify `ValueError: Message outside fixed timing
rounds/phases` in `mai_measurement.descriptor`. The released observer still used
`0 <= round <= 6`, inherited from the timing pilot. Its client boundary also
required Reference only. Both conflict with the already adopted 30-round main
jobs under Reference and Controlled Non-IID.

The prior screenshot/report printed evaluation through round 5. It is not a
complete internal progress ledger and the supplied traceback does not contain the
rejected numeric round. We do not infer the exact failed or last completed round
from the console alone. Repair preparation reads the retained B history and
records the last completed evaluation, without using its loss to select a result.

## Correction, not removal of validation

The observer obtains the permitted round budget and condition(s) from the
job-specific, digest-checked `fixed_config.json` embedded resource. In main jobs
this is exactly 30 rounds and the bound single condition. The timing-pilot default
still has its original six-round Reference-only contract; this correction does
not rewrite historical timing measurements or their warm-up definition.

The main path accepts initial evaluation at round 0, fit and regular evaluation at
rounds 1–30, and selected-checkpoint evaluation at any trained round 1–30. It
rejects round 31, invalid stages, conflicting conditions, invalid round types,
missing instruction records at the server send boundary, missing responses and
duplicate client requests. Existing intentional identity-negative client probes
still reach the inner identity guard before a data callback.

The byte-counting formula, clocks, training kernel, optimiser, model, loss,
checkpoint rule, thresholds, schedules and PRG primitive are unchanged. New scope
fields are local diagnostic JSON only; they are not added to model-update
messages. Main cost totals continue to include all planned rounds, including
round one; no timing-pilot warm-up exclusion is imported into the main analysis.

## Pre-training envelope gate

Every B/C job Probe now exercises the production server and client measurement
boundaries over the complete configured horizon with native installed Flower
Message/Metadata/Record objects and an explicitly local fixture transport. No
benchmark inputs, learned parameters, optimiser steps, cryptographic secrets or
live services are used. For a 30-round job this checks 192 request/response pairs
for B or 462 for C, including all four secure stage labels per round, initial
validation, the selected-model phase, and public handshake envelopes. Six invalid
cases must be rejected before their callback.

These are envelope/measurement checks, **not** execution of 30 training rounds,
SecAgg+, or Deployment Engine. All 20 main B/C job templates and their cloned
attempts undergo this gate before training is authorised. The previously added
all-A/B/C preflight-before-A-training rule remains in effect.

## Preserving the research record

`PrepareRepair` must use the reviewed immediate parent:
`/home/mai/EHR_FL_Main_v1/campaigns/20260909T071824Z_83a874db`.
The parent campaign/source/report hashes are fixed in
`reference/measurement_scope_fix/repair_policy.json`. The exact internal failure
and the local contiguous partial-B history are checked too. The parent is checked
again after preparation and is not modified. Original A and partial B remain
archived, incomplete-block execution evidence, not newly labelled pilot data.

The successor copies the frozen development tensors, initial states, draw
schedules and origins byte-for-byte; it does not regenerate or retune them, copy
trained checkpoints, or read test rows. No main training is performed by this
repair preparation. After preparation review, a new campaign-specific TRAIN
receipt is required before restarting the entire A/B/C first block. Neither the
old receipt nor `RetryBlock` is appropriate for this code defect. No best-of-old-
and-new selection is allowed.

## Failure diagnostics

If a future worker fails, the block exporter retains an allowlisted
`server_failure_capsule.json`: configuration, error class, known-safe observer
message, file/line/function frames and last recorded evaluation round where
available. Unknown messages, source-code lines, local variables, per-session
scores, model/update arrays and hashes, secrets, stores and full logs are not
exported. The original failure is preserved if diagnostic extraction itself fails.
Full logs remain locally available to the researcher.

## Validation limits

See the accompanying preparation QA and test logs. Local stand-in tests are not
native Flower/Deployment evidence. On the target host no failing or skipped unit
test is accepted. The new native envelope gates must also pass for every bound
B/C template. Passing this repair does not authorise test evaluation, certify
cryptographic privacy, or resolve missing dataset-publication/licensing records.
