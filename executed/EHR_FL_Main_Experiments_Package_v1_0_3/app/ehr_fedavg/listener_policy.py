"""Normalise numeric listener addresses without relaxing the local-only policy.

IPv4-mapped IPv6 is inspected explicitly. Python 3.12.3 must not be assumed to
classify a mapped address correctly through IPv6Address.is_loopback alone.
Only equivalents of the previously allowed endpoints (127.0.0.1 and ::1)
are accepted. Private networks, wildcards, hostnames and malformed addresses
remain blocked. No DNS queries or system changes are performed.
"""
from __future__ import annotations

import ipaddress
from typing import Any

POLICY_ID = 'exact-loopback-with-ipv4-mapped-normalisation-v1'


def classify_listener_address(address: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        'allowed': False,
        'normalized_address': None,
        'ipv4_mapped': False,
        'classification': 'invalid_numeric_address',
    }
    if not isinstance(address, str) or not address or address != address.strip():
        return result
    # psutil supplies an address and port separately. Do not accept URIs,
    # hostnames, bracketed endpoints, CIDRs or scoped addresses here.
    if '%' in address:
        result['classification'] = 'scoped_address_not_configured'
        return result
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return result
    mapped = parsed.ipv4_mapped if isinstance(parsed, ipaddress.IPv6Address) else None
    effective = mapped if mapped is not None else parsed
    result['normalized_address'] = str(effective)
    result['ipv4_mapped'] = mapped is not None
    if effective.is_unspecified:
        result['classification'] = 'wildcard_not_allowed'
    elif effective == ipaddress.IPv4Address('127.0.0.1'):
        result['allowed'] = True
        result['classification'] = 'ipv4_mapped_loopback' if mapped is not None else 'ipv4_loopback'
    elif effective == ipaddress.IPv6Address('::1'):
        result['allowed'] = True
        result['classification'] = 'ipv6_loopback'
    else:
        result['classification'] = 'not_a_configured_loopback_endpoint'
    return result
