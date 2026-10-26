# Attribution and modification notice

Flower source snapshots are Copyright 2025 Flower Labs GmbH, Apache License 2.0. The pinned HMAC-SHA256 helper originates from contributor `saurabhvmagdum`, Flower PR #7570, head commit `e87804ad00c46bff6dc3f058b73de81529e50bc0`. The proposal was not merged at verification. Its authorship is not attributed to the student or to this assistant.

`sources/backport_secaggplus_utils.py` is a **modified derivative** of the Flower 1.22.0 helper: its PRG function is replaced with the pinned contributor function and hashlib/hmac/struct imports are added. This derivative has not been endorsed or released by Flower. See reference/helper_change.diff and CHANGE_MANIFEST.json. Upstream licence notices are retained; a copy of Apache 2.0 accompanies the snapshots.

The local wheel, if successfully built on the host, has a distinguishable local version, updated RECORD/metadata and an additional provenance file. The only Flower runtime source file changed is the helper. No unrelated AGENTS.md change is carried over.

Validation policies and public toy oracles are derived from earlier project engineering packages and adapted for this bounded evaluation. They are not cryptographic proofs. No original code's security claims are adopted by copying a docstring.
