#!/usr/bin/env python3
"""Check (default) or explicitly regenerate the lossless FAB config carrier."""
from pathlib import Path
import argparse, hashlib

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('fixed_config.json','encoding_schema.json')


def render(app_directory: Path) -> str:
    texts={n:(app_directory/'ehr_fedavg'/n).read_bytes() for n in NAMES}
    s='"""Generated lossless public config resources for Flower 1.22 FABs.\n\nDo not edit manually. No session records, arrays, weights or credentials are embedded.\nRebuild/check using scripts/generate_bundled_resources.py.\n"""\n\nRESOURCE_BYTES = {\n'
    for n,b in texts.items():s+=f'    {n!r}: {b!r},\n'
    s+='}\n\nRESOURCE_SHA256 = '+repr({n:hashlib.sha256(b).hexdigest() for n,b in texts.items()})+'\n'
    return s


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write',action='store_true',help='Developer-only regeneration; changes the release and requires new manifest')
    args=ap.parse_args();p=ROOT/'app/ehr_fedavg/_embedded_resources.py';want=render(ROOT/'app').encode('utf-8')
    if args.write:p.write_bytes(want)
    elif not p.is_file() or p.read_bytes()!=want:raise SystemExit('STOP: generated resource does not match original source JSON')
    print('BUNDLED_RESOURCES_MATCH_ORIGINAL_JSON')
