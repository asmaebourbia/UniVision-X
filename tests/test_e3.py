"""Unit tests / offline smoke without pretrained downloads."""
from pathlib import Path
import tempfile
import torch
from univision.synthetic import make_dataset
from univision.e3_data import MeViSE3
from univision.e3_model import UniVisionE3


def make_fake_tokens(root):
    pooled=torch.load(Path(root)/'SYNTHETIC_TEXT_ONLY.pt',map_location='cpu',weights_only=True)
    result={}
    for key,val in pooled.items():
        result[key]={"tokens":val.unsqueeze(0).repeat(8,1).half(),
                     "attention_mask":torch.ones(8,dtype=torch.uint8)}
    out=Path(root)/'FAKE_TOKEN_CACHE_FOR_TEST_ONLY.pt'
    torch.save(result,out)
    return out


def test_e3_data_temporal_sample_and_backward():
    with tempfile.TemporaryDirectory() as d:
        make_dataset(d,image_size=64,n_train=2,n_val=2,nframes=6)
        tokenfile=make_fake_tokens(d)
        ds=MeViSE3(d,'valid_u',image_size=64,clip_length=3,train=False,token_cache=tokenfile)
        s=ds[0]
        assert s['video'].shape==(3,3,64,64)
        assert s['text_tokens'].shape==(8,512)
        assert s['text_attention_mask'].shape==(8,)
        m=UniVisionE3(encoder='toy',width=16,heads=4,pretrained=False,freeze_encoder=True,
                      attention_size=8)
        logits,pres,h=m(s['video'][None],s['text_tokens'][None],s['text_attention_mask'][None])
        assert logits.shape==(1,3,1,64,64)
        assert pres.shape==(1,3)
        target=s['mask'][None]
        present=target.flatten(2).any(-1).float()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)+\
             torch.nn.functional.binary_cross_entropy_with_logits(pres,present)
        loss.backward()
        assert m.cross_attention.in_proj_weight.grad is not None
        assert m.presence_head[-1].weight.grad is not None
        assert not m.encoder.layers[0][0].weight.requires_grad


def test_e3_attention_mask_effect():
    m=UniVisionE3(encoder='toy',width=16,pretrained=False,attention_size=8)
    m.eval()
    x=torch.rand(1,2,3,64,64)
    tok=torch.randn(1,8,512)
    mask=torch.ones(1,8,dtype=torch.bool)
    with torch.no_grad():
        a,_,_=m(x,tok,mask)
        mask[:,4:]=False
        b,_,_=m(x,tok,mask)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.equal(a,b)
