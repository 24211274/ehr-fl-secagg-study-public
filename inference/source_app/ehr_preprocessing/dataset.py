"""Local client/partition-specific memory-mapped arrays and dynamic batch padding.

No raw identifiers are supplied to the neural model. Metadata is returned separately
for lookup/debugging and never concatenated with event features or target arrays.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .io_utils import verify_artefacts, sha256_file
from .pipeline import ALLOWED_PARTITIONS, CLIENTS, SCHEMA_SHA, approved_schema


class SessionDataset(Dataset):
    def __init__(self, root: Path, client: str, partition: str,
                 session_ids: list[str] | None = None, verify: bool = True):
        if client not in CLIENTS or partition not in ALLOWED_PARTITIONS:
            raise ValueError('Development loader permits C1/C2/C3 and training/validation only')
        self.root, self.client, self.partition = Path(root), client, partition
        if verify:
            verify_artefacts(self.root)
        self.schema = approved_schema(self.root/'encoding_schema.json')
        summary = json.loads((self.root/'preprocessing_summary.json').read_text())
        if summary['schema_sha256'] != SCHEMA_SHA or sha256_file(self.root/'relative_time_scaler.json') != summary['scaler_sha256']:
            raise ValueError('Schema/scaler identity mismatch')
        d=self.root/client/partition
        self.features=np.load(d/'features.npy',mmap_mode='r',allow_pickle=False)
        self.categories=np.load(d/'categorical_targets.npy',mmap_mode='r',allow_pickle=False)
        self.categorical_valid=np.load(d/'categorical_valid.npy',mmap_mode='r',allow_pickle=False)
        self.offsets=np.load(d/'offsets.npy',mmap_mode='r',allow_pickle=False)
        self.lengths=np.load(d/'lengths.npy',mmap_mode='r',allow_pickle=False)
        with (self.root/'session_manifest.csv').open(encoding='utf-8',newline='') as f:
            rows=[r for r in csv.DictReader(f) if r['client']==client and r['partition']==partition]
        rows.sort(key=lambda r:int(r['session_index']))
        if [int(r['session_index']) for r in rows]!=list(range(len(rows))) or len(rows)!=len(self.lengths):
            raise ValueError('Invalid session index')
        self.ids=[r['session_id'] for r in rows]
        if len(set(self.ids))!=len(self.ids):
            raise ValueError('Duplicate session ID')
        n=len(self.features)
        if self.features.shape!=(n,119) or self.features.dtype!=np.float32 or self.categories.shape!=(n,13) or self.categories.dtype!=np.int64 or self.categorical_valid.shape!=(n,13) or self.categorical_valid.dtype!=np.bool_:
            raise ValueError('Artefact shape or dtype mismatch')
        if self.offsets.shape!=(len(self.lengths)+1,) or self.offsets[0]!=0 or self.offsets[-1]!=n or not np.array_equal(np.diff(self.offsets),self.lengths) or np.any(self.lengths<1):
            raise ValueError('Invalid ragged-array offsets')
        mapping={sid:i for i,sid in enumerate(self.ids)}
        if session_ids is None:
            self.indices=list(range(len(self.ids)))
        else:
            if any(s not in mapping for s in session_ids):
                raise ValueError('Session requested from wrong client/partition or absent')
            # Replacement draw schedules may intentionally request an ID multiple times.
            self.indices=[mapping[s] for s in session_ids]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        j=self.indices[index]
        lo,hi=int(self.offsets[j]),int(self.offsets[j+1])
        x=torch.from_numpy(np.array(self.features[lo:hi],dtype=np.float32,copy=True))
        cat=torch.from_numpy(np.array(self.categories[lo:hi],dtype=np.int64,copy=True))
        valid=torch.from_numpy(np.array(self.categorical_valid[lo:hi],dtype=bool,copy=True))
        return {'x':x,'categorical':cat,'categorical_valid':valid,
                'binary':x[:,110:113].clone(),'temporal':x[:,113:119].clone(),
                'session_id':self.ids[j], 'client':self.client, 'partition':self.partition}


def collate_sessions(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError('Cannot collate an empty batch')
    lengths=torch.tensor([len(s['x']) for s in samples],dtype=torch.int64)
    if (lengths<1).any():
        raise ValueError('Empty input session')
    width=samples[0]['x'].shape[-1]
    if width!=119:
        raise ValueError('Approved width is 119')
    b,t=len(samples),int(lengths.max())
    x=torch.zeros((b,t,width),dtype=torch.float32)
    cat=torch.full((b,t,13),-100,dtype=torch.int64)
    valid=torch.zeros((b,t,13),dtype=torch.bool)
    binary=torch.zeros((b,t,3),dtype=torch.float32)
    temporal=torch.zeros((b,t,6),dtype=torch.float32)
    for i,s in enumerate(samples):
        n=int(lengths[i])
        expected={'x':(n,119),'categorical':(n,13),'categorical_valid':(n,13),
                  'binary':(n,3),'temporal':(n,6)}
        if any(tuple(s[k].shape)!=v for k,v in expected.items()):
            raise ValueError('Inconsistent sample dimensions')
        if not torch.isfinite(s['x']).all():
            raise ValueError('Nonfinite sample')
        x[i,:n]=s['x'];cat[i,:n]=s['categorical'];valid[i,:n]=s['categorical_valid']
        binary[i,:n]=s['binary'];temporal[i,:n]=s['temporal']
    mask=torch.arange(t)[None,:]<lengths[:,None]
    return {'x':x,'categorical':cat,'categorical_valid':valid,
            'binary':binary,'temporal':temporal,'lengths':lengths,'padding_mask':mask,
            'metadata':[{'session_id':s['session_id'],'client':s['client'],'partition':s['partition']} for s in samples]}
