# MAIN-FIXTURE-IMPORT-002 — test-fixture resource isolation

Release 1.0.2 corrects the four native template-copy-reprobe tests introduced in
1.0.1. The last target report contains 488 tests: 484 passed, four errors, no
failures or skips. All four native tests failed at the strict source-resource
comparison, before their native FAB build, with `Original JSON and bundled
representation differ: fixed_config.json`.

## Cause and bounded correction
The unittest process imported `ehr_fedavg.bundle_checks` and its resource module
from the release's default `app/`. The tests then used `bind_app` to write a
30-round, seed-11 main configuration into a temporary application. The strict
checker still referenced the release-default in-memory carrier. Changing the
filesystem or prepending `sys.path` cannot replace a cached import.

The four tests now run each native preflight in a fresh interpreter (`-I -B`),
binding the temporary app path before any application import. The test helper
checks the imported resource/carrier/checker paths and configuration hashes.
It invokes the original native FAB builder and isolated callback loader. The
existing test-only choice to omit benchmark-array loading remains explicit:
these fixtures contain no real dataset. The actual PrepareRepair worker retains
all full client-data checks and probes all twenty B/C templates and their copies.

The resource-only regression tests reproduce the prior mismatch with the real
project resource reader, and confirm isolation across both configurations,
conditions and several seeds. They confirm that changed JSON, changed carrier
bytes, a broken schema link and a missing carrier remain errors. They also check
that a poisoned PYTHONPATH is ignored and the unittest process keeps its module
objects unchanged. No sys.modules clearing, importlib.reload trick, disabled
hash check or rewritten expected value is used.

## Unchanged components
All files in `app/`, `main_study/`, `scripts/`, `config/`, `protocol/`,
`validation_support/`, `fullsize_support/` and `timing_support/` are unchanged
from release 1.0.1. No model, loss, training, calibration, sampling, PRG,
authorisation, strict bundle verification or exclusive-directory code changes.
The 1.0.1 campaign-manifest format stays unchanged; source-release identity is
its separately hashed PACKAGE_SHA256SUMS.txt, not the manifest-format version.
Only the existing repair policy gains an additive test-harness amendment so the
new campaign's amendment includes this failed preparation and corrective scope.

## Evidence preservation and next action
Run PrepareRepair against the original reviewed parent campaign
`20260909T040230Z_5ecc549d`, not the incomplete repair preparation
`20260909T060409Z_1b57116a`. Preserve both. The original parent's frozen inputs,
initial checkpoints and schedules are reused byte-for-byte. The completed A
from the partial first block stays archived; this preparation neither repeats
it nor uses its trained weights. All previous campaign-bound training receipts
remain invalid for the successor. No training or test access is authorised by
this release.

## Verification boundaries
A passing resource-only test is not a native Flower build. The four native
clone/reprobe tests remain conditional on Flower in the local preparation
container and mandatory without skips on the target host. No surrogate Flower
package is installed or passed off as the real dependency. Local test logs and
source-transition evidence record precisely what was run. There is no new claim
of privacy certification, public dataset availability or detection performance.

Official Python references supporting the import explanation:
https://docs.python.org/3.12/reference/import.html#the-module-cache
https://docs.python.org/3.12/using/cmdline.html#cmdoption-I
