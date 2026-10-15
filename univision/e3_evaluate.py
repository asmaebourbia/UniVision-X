"""Full-video MeViS E3 local evaluation; masks are NOT official leaderboard scores."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from .e3_data import MeViSE3
from .e3_model import UniVisionE3
from .metrics import mask_iou,mask_dice,boundary_f
from .utils import available_device,write_json,ensure_dir

@torch.inference_mode()
def predict(model,video,tokens,mask,chunk=3,amp=True,presence_threshold=0.5):
    dev=next(model.parameters()).device
    state=None;all_masks=[];all_presence=[]
    for start in range(0,len(video),chunk):
        x=video[start:start+chunk][None].to(dev)
        tt=tokens[None].to(dev)
        tm=mask[None].to(dev)
        with torch.autocast(device_type=dev.type,enabled=amp and dev.type=='cuda'):
            logit,presence,state=model(x,tt,tm,state)
        pred=(logit.float().sigmoid()>0.5)
        is_present=presence.float().sigmoid()>presence_threshold
        pred=pred & is_present[:,:,None,None,None]
        all_masks.append(pred[0,:,0].cpu().numpy())
        all_presence.append(is_present[0].cpu().numpy())
        state=state.detach()
    return np.concatenate(all_masks,axis=0),np.concatenate(all_presence)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--token-cache',required=True)
    p.add_argument('--split',default='valid_u')
    p.add_argument('--out',default='outputs/E3_eval_clean')
    p.add_argument('--corruption',choices=['clean','noise','blur','fog','dark'],default='clean')
    p.add_argument('--chunk-size',type=int,default=3)
    p.add_argument('--presence-threshold',type=float,default=0.5)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--save-png',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    dev=available_device()
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    conf=ckpt['config']
    model=UniVisionE3(encoder=conf['encoder'],width=conf['width'],pretrained=False,
                      freeze_encoder=conf['freeze_encoder']).to(dev).eval()
    model.load_state_dict(ckpt['model'])
    ds=MeViSE3(args.data,args.split,image_size=conf['image_size'],full_video=True,train=False,
               corrupt_mode=args.corruption,token_cache=args.token_cache,limit=args.limit)
    ensure_dir(args.out)
    rows=[];tot_empty=0;falsepos=0;tot_present=0;j_present=[]
    for i in tqdm(range(len(ds)),desc='full video E3'):
        sample=ds[i]
        start=time.perf_counter()
        pred,presence=predict(model,sample['video'],sample['text_tokens'],sample['text_attention_mask'],
                              args.chunk_size,args.amp,args.presence_threshold)
        if dev.type=='cuda':torch.cuda.synchronize()
        duration=time.perf_counter()-start
        gt=sample['mask'][:,0].numpy()>0.5
        per_j=[mask_iou(p,g) for p,g in zip(pred,gt)]
        per_d=[mask_dice(p,g) for p,g in zip(pred,gt)]
        per_f=[boundary_f(p,g) for p,g in zip(pred,gt)]
        for p,g,j in zip(pred,gt,per_j):
            if g.any():tot_present+=1;j_present.append(j)
            else:tot_empty+=1;falsepos+=int(p.any())
        rec={'video':sample['name'],'exp_id':sample['exp_id'],'text':sample['text'],
             'frames':len(pred),'J':float(np.mean(per_j)),'Dice':float(np.mean(per_d)),
             'boundary_F_proxy':float(np.mean(per_f)),
             'J_and_proxy_F':float(np.mean([(j+f)/2 for j,f in zip(per_j,per_f)])),
             'fps_model_only':len(pred)/max(duration,1e-6)}
        rows.append(rec)
        if args.save_png:
            folder=Path(args.out)/'Annotations'/sample['name']/sample['exp_id']
            ensure_dir(folder)
            oh,ow=sample['orig_hw']
            for mask,frame in zip(pred,sample['frame_names']):
                Image.fromarray(mask.astype('uint8')*255).resize((ow,oh),Image.Resampling.NEAREST).save(folder/f'{frame}.png')
        if (i+1)%20==0:print(f'Processed {i+1}/{len(ds)}',flush=True)
    means={name:float(np.mean([row[name] for row in rows])) for name in
           ['J','Dice','boundary_F_proxy','J_and_proxy_F','fps_model_only']}
    summary={'split':args.split,'corruption':args.corruption,'n_expressions':len(ds),
             'local_metrics':means,'present_frame_J_mean':float(np.mean(j_present)) if j_present else None,
             'absent_frames':tot_empty,'false_positive_absent_frames':falsepos,
             'false_positive_absent_rate':falsepos/max(tot_empty,1),
             'presence_threshold':args.presence_threshold,
             'warning':'Resized local J/Dice and boundary proxy; NOT official MeViS J&F. FPS excludes image load.'}
    write_json(summary,Path(args.out)/'summary.json')
    with open(Path(args.out)/'per_expression.csv','w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
