"""Dimension-parameterised LSTM core from the report; no final encoder is assumed."""
from __future__ import annotations
import hashlib
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def set_training_seed(seed: int) -> None:
    """Training randomness only. Never use this to derive cryptographic keys."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def state_digest(model: nn.Module) -> str:
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class CompactLSTMAutoencoder(nn.Module):
    def __init__(self, categorical_sizes: list[int]):
        super().__init__()
        if len(categorical_sizes)!=13 or any(k<1 for k in categorical_sizes):
            raise ValueError('Thirteen positive categorical head sizes required')
        self.categorical_sizes=tuple(categorical_sizes)
        self.input_dim=sum(categorical_sizes)+9
        # Fixed constructor order is part of the initialisation definition.
        self.encoder=nn.LSTM(self.input_dim,64,num_layers=1,batch_first=True,
                             dropout=0.0,bidirectional=False)
        self.to_latent=nn.Linear(64,32)
        self.decoder=nn.LSTM(32,64,num_layers=1,batch_first=True,
                             dropout=0.0,bidirectional=False)
        self.initial_h=nn.Linear(32,64)
        self.initial_c=nn.Linear(32,64)
        self.output_dropout=nn.Dropout(.1)
        self.categorical_heads=nn.ModuleList([nn.Linear(64,k) for k in categorical_sizes])
        self.binary_head=nn.Linear(64,3)
        self.temporal_head=nn.Linear(64,6)

    def forward(self,x: torch.Tensor,lengths: torch.Tensor):
        if x.ndim!=3 or x.shape[-1]!=self.input_dim or lengths.shape!=(x.shape[0],):
            raise ValueError('Input/length shape mismatch')
        if (lengths<1).any() or (lengths>x.shape[1]).any() or not torch.isfinite(x).all():
            raise ValueError('Invalid sequence lengths or nonfinite values')
        packed=pack_padded_sequence(x,lengths.detach().cpu(),batch_first=True,enforce_sorted=False)
        _,(h,_) = self.encoder(packed)
        z=torch.tanh(self.to_latent(h[-1]))
        decoder_x=z[:,None,:].expand(-1,x.shape[1],-1)
        out,_=self.decoder(decoder_x,(self.initial_h(z)[None,:,:],self.initial_c(z)[None,:,:]))
        out=self.output_dropout(out)
        return [head(out) for head in self.categorical_heads],self.binary_head(out),self.temporal_head(out)


def session_losses(outputs,cat_targets,binary_targets,temporal_targets,lengths,cat_valid=None):
    cats,binary,time=outputs
    b,t,_=binary.shape
    if cat_targets.shape!=(b,t,13) or binary_targets.shape!=(b,t,3) or temporal_targets.shape!=(b,t,6):
        raise ValueError('Target shape mismatch')
    if cat_valid is None:
        cat_valid=cat_targets>=0
    if cat_valid.shape!=cat_targets.shape:
        raise ValueError('Categorical validity mask shape mismatch')
    categorical_sum=binary.new_zeros((b,t))
    for j,logits in enumerate(cats):
        # Unknown, missing and training-unseen targets are masked, not penalised as threats.
        target=cat_targets[:,:,j].clone().masked_fill(~cat_valid[:,:,j],-100)
        categorical_sum+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),
                            reduction='none',ignore_index=-100,label_smoothing=0).reshape(b,t)
    categorical_n=cat_valid.sum(-1)
    cat_loss=categorical_sum/categorical_n.clamp_min(1)
    bin_loss=F.binary_cross_entropy_with_logits(binary,binary_targets,reduction='none').mean(-1)
    temp_loss=F.smooth_l1_loss(time,temporal_targets,reduction='none',beta=1.0).mean(-1)
    n_groups=2+(categorical_n>0).to(binary.dtype)
    event_loss=(cat_loss+bin_loss+temp_loss)/n_groups
    mask=torch.arange(t,device=lengths.device)[None,:]<lengths[:,None]
    return (event_loss*mask).sum(-1)/lengths, event_loss, mask


def new_optimiser(model):
    return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,
              betas=(.9,.999),eps=1e-8,amsgrad=False,foreach=False,fused=False)
