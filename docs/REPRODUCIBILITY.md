# Reproduction and verification

## Public archive integrity

From the repository root:

```console
python -I -B tools/verify_public_archive.py
```

This checks the exact public file set and SHA-256 values without reading data or running experiments. The public executed-source tree is a curated subset; its original manifest describes the complete private package, not a claim that excluded files are present publicly.

## Synthetic arithmetic checks

Use Python 3.12 or later in a new environment. The audit's recorded analysis versions were NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0. Install these analysis dependencies, then:

```console
cd analysis/final_score_audit
python -B -m unittest -v test_audit_metrics
```

These 12 tests use synthetic in-memory arrays. They check AP, AUROC, confusion counts, ties, undefined metrics and score transformations. They do not require benchmark access or a model.

## Full stored-score audit and training replay

The actual saved-score audit requires five controlled input ZIPs. Download them with authorised access and follow [the complete archive instructions](https://github.com/24211274/ehr-fl-secagg-study/blob/study-artifacts-v1.30/docs/AVAILABILITY.md). The original audit code is unchanged. Full training additionally requires controlled benchmark/prepared inputs and runtime packages; historical path bindings must be mapped into a new workspace. No new benchmark training was performed during publication.

## Session inference

Use [the hosted application](https://session-check-demo.onrender.com) and [its guide](user_guides/Session_Check_User_Guide_EN.pdf). Public source under `inference/` documents the application and includes synthetic samples. It excludes model and fitted assets. To run locally, obtain the complete `inference/` directory from the controlled archive and use its quick start. The public source checkout alone is not a complete local model bundle.
