# MAIN-ATTEMPT-CLONE-001: execution-staging amendment

## Evidence and diagnosis
The reviewed MAIN-11-reference attempt_1 completed A (30 epochs, 5,100 updates) then failed during B Probe. B/probe logs identify FileExistsError at bundle_preflight/reference. The frozen templates have already undergone native FAB probing. The old copytree ignore list omitted the resulting bundle_preflight tree. Copying it made the next exclusive mkdir fail. This is a source-code defect, not a transient infrastructure exception and not evidence of a bad model.

## Corrective scope
`stage_attempt_job` copies only apps, client_data, runtime.json and job.json, discarding only generated FAB/bytecode files within application inputs. It retains the template, refuses existing targets, refuses trained/unknown root artefacts and symlink inputs, creates a fresh run token and uses the same project writer to relocate paths and bind rounds/callbacks. It does not modify Flower's bundle checker or relax its exclusive directory creation.

`execute_block` calls this same helper, completes all three Probe actions, then starts the unchanged Train workers in the adopted order. Preparation tests the exact successful-template-probe -> clone -> probe lifecycle for every federated job, before authorisation for training. The all-cell preflight changes external staging order only, not a measured training/aggregation workflow.

## Frozen-campaign handling
PrepareRepair verifies the exact reviewed parent campaign, source manifest, preparation report, failed block report, state of progress and archived A artefact hashes. The fixed source is copied into a fresh successor campaign. Frozen development arrays, preprocessing constants, initial checkpoints and schedules are copied byte-for-byte, without hard-linking back to the parent or re-fitting/regenerating. The thirty job configs are re-materialised and tested; no trained checkpoint from the partial block is used.

The parent is read only. Its archived A is a real main training outcome, not retrospectively relabelled a pilot. It remains documented, but is not a candidate for cherry-picking against a later A. After review, the entire block will be restarted with the same model, data, seeds and budget under a new campaign-bound receipt; partial cells are not spliced together. No main training or final test access is performed by PrepareRepair. Old receipts cannot authorise the successor because campaign/source digests differ. All failures remain preserved. This is one explicit code correction, not a hidden reset of transient retry counters.

## Unchanged scientific implementation
All application/ML/security source files, parameter initialisation/loss/optimiser code, evaluator and calibration code, protocol decisions, data definitions, schedules rules, timing scopes and cryptographic PRG are byte-identical to v1.0. The new source has a separately verified manifest. This amendment is made after observing A's routine-validation curve; no parameter or outcome-dependent selection was made using that curve. Final test metrics remain unseen in this workflow; previously declared structural exposure/provenance limitations remain unchanged.

## Testing scope
The local regression tests reproduce the former failure for B and C with real directories, then exercise the corrected copy, relocation, immutable parent evidence, successor sealing and all-probes-before-training order. Where stated, workers and protocol execution are mocked. Four native tests use the actual patched Flower builder and isolated application loader but no benchmark datasets; they are mandatory on the target host. The target PrepareRepair additionally probes the actual full training/validation job inputs after copying all twenty B/C templates. Passing unit tests does not assert Deployment or cryptographic security.

## Source references
Python 3.12 shutil.copytree copies a directory recursively; an ignore callback selects exclusions. Existing targets are not silently overwritten in this fix. Python 3.12 Path.mkdir(exist_ok=False) raises FileExistsError for an existing directory. References: https://docs.python.org/3.12/library/shutil.html and https://docs.python.org/3.12/library/pathlib.html. Project-specific diagnosis is grounded in the uploaded failed block and released source, not inferred from these generic references.


## Follow-up test harness amendment in release 1.0.2
The initial code repair's native fixture tests used the wrong import context.
MAIN-FIXTURE-IMPORT-002 (see FIXTURE_IMPORT_FIX.md) isolates these tests without
changing any production code described above. The additive repair-policy field
records this follow-up and the previous failed preparation; no new training
or test authorisation is implied. The original parent remains the source of
the frozen inputs.
