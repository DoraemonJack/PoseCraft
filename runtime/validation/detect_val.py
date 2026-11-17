# Adapted paths/job limit only from frozen detect_projection_clean_v2.py; source SHA256 806e9d609310b2671c70ee908118428d135a0b20f5ab83adf83f09011d133394
"""Bounded, resumable RGB-D detector; provisional outputs, not training labels."""
import os,sys,json,inspect,hashlib,time,traceback
from pathlib import Path
import numpy as np,torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
W=Path('/path/to/research100/grounded_sam2_reference_20261101_v1');sys.path.insert(0,str(W/'sam2_official'));from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sys.path.insert(0,'/path/to/research100/strict_real30_image_modules_20261102_v1');from image_reference_projection_v1 import project_instances
ROOTS=[Path('/path/to/research300/posecraft_val_reference_correction_20261117/frames')];O=Path('/path/to/research300/posecraft_val_reference_correction_20261117/detector');O.mkdir(exist_ok=True)
def scene_root(scene):
 for root in ROOTS:
  if (root/scene).is_dir():return root
 return None
def main():
 device='cuda:0';torch.set_num_threads(1)
 jobs=[json.loads(x) for x in (O.parent/'new_jobs.jsonl').read_text(encoding='utf8').splitlines()];splitfile=O/'split_between_candidates_v1.jsonl'
 if splitfile.exists():
  for line in splitfile.read_text(encoding='utf8').splitlines():
   z=json.loads(line);jobs.append(dict(scene=z['scene'],phrase=z['parent_phrase'],detector_phrase=z['detector_phrase'],kind=z['kind'],cleanup_status=z['cleanup_status']))
 selected={};missing=set();review=0
 for j in jobs:
  if j['cleanup_status']!='detector_candidate':review+=1;continue
  root=scene_root(j['scene'])
  if root is None:missing.add(j['scene']);continue
  key=(j['scene'],j['detector_phrase']);entry=selected.setdefault(key,dict(scene=key[0],detector_phrase=key[1],kinds=set(),source_phrases=set()));entry['kinds'].add(j['kind']);entry['source_phrases'].add(j['phrase'])
 queue=[]
 for key,j in sorted(selected.items()):
  if int(hashlib.sha256((key[0]+'|'+key[1]).encode()).hexdigest(),16)%4 != int(os.environ['WORKER_ID']):continue
  imgs=sorted((scene_root(j['scene'])/j['scene']/'color').glob('*.jpg'))
  imgs=[imgs[i] for i in sorted(set(round(k*(len(imgs)-1)/3) for k in range(4)))] if imgs else []
  for rgb in imgs:
   stem=rgb.stem;folder=scene_root(j['scene'])/j['scene'];
   if not (folder/'depth'/f'{stem}.png').exists() or not (folder/'pose'/f'{stem}.txt').exists():continue
   token=hashlib.sha256((j['scene']+'|'+stem+'|'+j['detector_phrase']).encode()).hexdigest()[:24]
   if not (O/(token+'.json')).exists():queue.append((j,stem,token))
 max_jobs=int(os.environ.get('REFERENCE_MAX_JOBS','100000'));queue=queue[:max_jobs]
 print(json.dumps(dict(available_scene_phrase=len(selected),missing_scenes=len(missing),review_jobs=review,batch_jobs=len(queue))),flush=True)
 if not queue:return
 dp=AutoProcessor.from_pretrained(W/'dino_base',local_files_only=True);dm=AutoModelForZeroShotObjectDetection.from_pretrained(W/'dino_base',local_files_only=True).to(device).eval();sp=SAM2ImagePredictor(build_sam2('configs/sam2/sam2_hiera_s.yaml',str(W/'sam2_original_checkpoint.pt'),device=device,mode='eval'));cache={};done=0
 for j,stem,token in queue:
  started=time.monotonic();row=dict(scene=j['scene'],frame=stem,detector_phrase=j['detector_phrase'],kinds=sorted(j['kinds']),source_phrases=sorted(j['source_phrases']),projection_provisional=True,calibration_verified=False,not_training_ready=True)
  try:
   fk=(j['scene'],stem);folder=scene_root(j['scene'])/j['scene']
   if fk not in cache:cache[fk]=(np.array(Image.open(folder/'color'/f'{stem}.jpg').convert('RGB'),copy=True),np.asarray(Image.open(folder/'depth'/f'{stem}.png'),float)*.001,np.loadtxt(folder/'pose'/f'{stem}.txt'),np.loadtxt(folder/'intrinsics_depth.txt')[:3,:3],np.loadtxt(folder/'intrinsics_color.txt')[:3,:3])
   rgb,depth,pose,kd,kc=cache[fk];im=Image.fromarray(rgb);x=dp(images=im,text=j['detector_phrase']+'.',return_tensors='pt').to(device)
   with torch.inference_mode():y=dm(**x)
   kw=dict(text_threshold=.25,target_sizes=[im.size[::-1]]);kw['threshold' if 'threshold' in inspect.signature(dp.post_process_grounded_object_detection).parameters else 'box_threshold']=.3;d=dp.post_process_grounded_object_detection(y,x.input_ids,**kw)[0];boxes=d['boxes'].cpu().numpy();scores=d['scores'].cpu().numpy();ix=np.argsort(-scores)[:8];boxes,scores=boxes[ix],scores[ix];masks=[]
   if len(boxes):
    sp.set_image(rgb)
    for box in boxes:
     with torch.inference_mode():mask,_,_=sp.predict(box=box,multimask_output=False)
     masks.append(mask[0].astype(bool))
   masks=np.asarray(masks,bool).reshape(-1,im.height,im.width);detections=project_instances(masks,depth,kd,kc,pose,color_from_depth=None,calibration_verified=False);packed={'masks':masks,'boxes':boxes,'scores':scores}
   for i,det in enumerate(detections):
    for k,v in det.items():
     a=np.asarray(v)
     if a.dtype.kind in 'biuf' and a.size:packed[f'projection_{i}_{k}']=a
   np.savez_compressed(O/(token+'.npz'),**packed);row.update(status='detected' if len(boxes) else 'no_detection',detections=len(boxes),projected_instances=len(detections))
  except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e)[:300])
  row['seconds']=time.monotonic()-started;(O/(token+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2));done+=1;print(json.dumps(dict(done=done,total=len(queue),**row),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
