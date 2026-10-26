"""Exact candidate admission for FAB checks; no general local-version allowance.

The approved downstream version differs from the upstream label deliberately.
Version identity alone is insufficient: use the existing installed-file,
function and client/server binding verification before calling the FAB builder.
"""
from __future__ import annotations
from mai_runtime_identity import (VERSION, FUNCTION_SHA256, HELPER_SHA256,
                                  EXPECTED_BUILD_ID, IdentityMismatch, binding_identity)


def require_fab_runtime(observed_version: str) -> dict:
    """Accept only the explicitly approved, source-verified local build."""
    if observed_version != VERSION:
        raise IdentityMismatch(
            f"Expected exact approved Flower build {VERSION} for pinned FAB semantics; "
            f"observed {observed_version!r}; no fallback")
    identity = binding_identity()
    expected = {
        "status": "PASS", "version": VERSION, "build_id": EXPECTED_BUILD_ID,
        "function_sha256": FUNCTION_SHA256, "helper_sha256": HELPER_SHA256,
        "client_server_bindings_match": True, "official_release": False,
        "remote_attestation": False, "privacy_claim_validated": False,
    }
    if not isinstance(identity, dict):
        raise IdentityMismatch("FAB runtime identity returned no verified record")
    for key, value in expected.items():
        actual = identity.get(key)
        if type(actual) is not type(value) or actual != value:
            raise IdentityMismatch("FAB runtime identity mismatch: " + key)
    return dict(identity)
