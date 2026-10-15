"""UniVision-X E3 train: cross-attention, presence-aware loss, uniform-time validation."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .data import normal_text
from .e3_data import MeViSE3
from .e3_model import UniVisionE3
from .metrics import segmentation_loss
from .utils import seed_everything, available_device, ensure_dir, num_parameters, write_json, worker_seed


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--token-cache',required=True)
    p.add_argument('--out',default='outputs/E3_fastvit_t8')
    p.add_argument('--encoder',default='fastvit_t8.apple_in1k')
    p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--clip-length',type=int,default=3)
    p.add_argument('--width',type=int,default=48)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--presence-weight',type=float,default=0.5)
    p.add_argument('--corruption-probability',type=float,default=0.3)
    p.add_argument('--grad-accum',type=int,default=2)
    p.add_argument('--freeze-encoder',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--limit-train',type=int,default=0)
    p.add_argument('--limit-val',type=int,default=0)
    p.add_argument('--resume',default=None)
    p.add_argument('--no-pretrained',action='store_true')
    return p.parse_args()


def make_model(args,pretrained):
    return UniVisionE3(encoder=args.encoder,pretrained=pretrained,
                       width=args.width,freeze_encoder=args.freeze_encoder)


def batch_step(model,batch,device,amp,presence_weight):
    x=batch['video'].to(device,non_blocking=True)
    m=batch['mask'].to(device,non_blocking=True)
    tokens=batch['text_tokens'].to(device,non_blocking=True)
    valid=batch['text_attention_mask'].to(device,non_blocking=True)
    present=m.flatten(2).any(-1).to(dtype=torch.float32)
    with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
        logits,pres_logit,_=model(x,tokens,valid)
        seg_loss=segmentation_loss(logits,m)
        presence_loss=F.binary_cross_entropy_with_logits(pres_logit.float(),present)
        loss=seg_loss+presence_weight*presence_loss
    return loss,seg_loss.detach(),presence_loss.detach()


@torch.no_grad()
def validate(model,loader,device,amp):
    model.eval()
    expression_j=[]
    present_j=[]
    empty_n=0; false_positive_n=0; mask_pixels_fp=0; empty_total_pixels=0
    presence_correct=0;presence_total=0
    for batch in tqdm(loader,desc='val E3',leave=False):
        x=batch['video'].to(device)
        y=batch['mask'].to(device)>0.5
        tokens=batch['text_tokens'].to(device)
        mask=batch['text_attention_mask'].to(device)
        with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
            logits,pres,_=model(x,tokens,mask)
        pred=(logits.float().sigmoid()>=0.5) & (pres.float().sigmoid()[:,:,None,None,None]>=0.5)
        for b in range(pred.shape[0]):
            val=[]
            for t in range(pred.shape[1]):
                p=pred[b,t,0];g=y[b,t,0]
                inter=(p&g).sum().item()
                union=(p|g).sum().item()
                score=1.0 if union==0 else inter/union
                val.append(score)
                if g.any():
                    present_j.append(score)
                else:
                    empty_n+=1
                    false_positive_n+=int(p.any().item())
                    mask_pixels_fp+=int(p.sum().item())
                    empty_total_pixels+=p.numel()
                presence_correct += int((pres[b,t].sigmoid() >= 0.5).item()==g.any().item())
                presence_total+=1
            expression_j.append(float(np.mean(val)))
    fp=false_positive_n/max(empty_n,1)
    return dict(val_macro_j=float(np.mean(expression_j)),
                val_present_frame_j=float(np.mean(present_j)) if present_j else 0.0,
                val_false_positive_absent_rate=fp,
                val_false_positive_pixel_rate=mask_pixels_fp/max(empty_total_pixels,1),
                val_presence_accuracy=presence_correct/max(presence_total,1),
                val_expressions=len(expression_j),
                val_absent_frames=empty_n)


def main():
    args=parse_args()
    if args.grad_accum < 1:raise ValueError('grad-accum must be >=1')
    seed_everything(args.seed)
    ensure_dir(args.out)
    config_path=Path(args.out)/'config.json'
    if args.resume and config_path.exists():
        old=json.loads(config_path.read_text())
        protected=['encoder','width','image_size','clip_length','freeze_encoder','token_cache']
        for key in protected:
            if old.get(key)!=getattr(args,key):
                raise ValueError(f'Cannot resume: {key} changed from {old.get(key)} to {getattr(args,key)}')
    write_json(vars(args),config_path)
    dev=available_device()
    train_ds=MeViSE3(args.data,'train',image_size=args.image_size,clip_length=args.clip_length,
                     train=True,corruption_probability=args.corruption_probability,
                     token_cache=args.token_cache,limit=args.limit_train)
    val_ds=MeViSE3(args.data,'valid_u',image_size=args.image_size,clip_length=args.clip_length,
                   train=False,corrupt_mode='clean',token_cache=args.token_cache,limit=args.limit_val)
    tr=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                  pin_memory=dev.type=='cuda',worker_init_fn=worker_seed)
    va=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                  pin_memory=dev.type=='cuda')
    model=make_model(args,pretrained=not args.no_pretrained).to(dev)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=args.lr,weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp and dev.type=='cuda')
    start_epoch=0;best=-float('inf');history=[]
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch=int(checkpoint['epoch'])+1
        best=float(checkpoint.get('best',best))
        if (Path(args.out)/'history.json').exists():
            history=json.loads((Path(args.out)/'history.json').read_text())
    print('E3 device:',dev,'params:',num_parameters(model),'train/val:',len(train_ds),len(val_ds),flush=True)
    for epoch in range(start_epoch,args.epochs):
        start=time.time()
        model.train();optimizer.zero_grad(set_to_none=True)
        total=0.0; seg_total=0.0;pres_total=0.0
        for idx,batch in enumerate(tqdm(tr,desc=f'E3 epoch {epoch+1}',leave=False)):
            loss,seg_loss,pres_loss=batch_step(model,batch,dev,args.amp,args.presence_weight)
            scaler.scale(loss/args.grad_accum).backward()
            if (idx+1)%args.grad_accum==0 or idx+1==len(tr):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            total+=float(loss.detach());seg_total+=float(seg_loss);pres_total+=float(pres_loss)
        metrics=validate(model,va,dev,args.amp)
        # Prespecified trade-off: present-target mask quality vs any-mask false positives.
        score=metrics['val_present_frame_j']-0.1*metrics['val_false_positive_absent_rate']
        rec={'epoch':epoch+1,'train_loss':total/max(len(tr),1),
             'train_seg_loss':seg_total/max(len(tr),1),'train_presence_loss':pres_total/max(len(tr),1),
             **metrics,'selection_score':score,'seconds':round(time.time()-start,1)}
        history.append(rec)
        print(json.dumps(rec),flush=True)
        ckpt={'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
              'scaler':scaler.state_dict(),'best':max(best,score),'config':vars(args)}
        torch.save(ckpt,Path(args.out)/'last.pt')
        if score>best:
            best=score
            torch.save(ckpt,Path(args.out)/'best.pt')
        write_json(history,Path(args.out)/'history.json')
    print('E3 finished',flush=True)

if __name__=='__main__':
    main()
