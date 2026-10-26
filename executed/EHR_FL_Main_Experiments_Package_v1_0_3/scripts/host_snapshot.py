#!/usr/bin/env python3
"""Executed with original project interpreter, only reads inventory."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'validation_support'))
from prg_validation.snapshot import snapshot
Path(sys.argv[1]).write_text(json.dumps(snapshot(),indent=2,sort_keys=True,allow_nan=False)+'\n')
