"""Frozen DINO + official SAM 2 observation-only support-surface observations.

Stores original scene vertex IDs, so the same IDs remain valid after the
verified rigid policy-frame transform. No semantic/instance/placement labels.
All imported camera projections explicitly retain the RGB-D extrinsic identity
approximation; geometric nearest-surface agreement is measured, not assumed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
MODEL_ROOT=Path('/path/to/research100/grounded_sam2_reference_20261101_v1')
OLD_CACHE=Path('/path/to/research300/refineplace_reference_clean_v2_batch_20261102_v1')
MESH=Path('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json')
LOCK=Path('/path/to/research100/official_split_training_audit_20261102_v1/official563_training_split_lock.json')
LABELS=['floor','table','desk','counter','cabinet','shelf','bed','chair','sofa','bench','stool']
PROMPT='floor . table . desk . countertop . cabinet . shelf . bed . chair . sofa . bench . stool .'
ALIASES={'floor':'floor','ground':'floor','table':'table','coffee table':'table','dining table':'table',
 'desk':'desk','counter':'counter','countertop':'counter','kitchen counter':'counter',
 'cabinet':'cabinet','kitchen cabinet':'cabinet','dresser':'cabinet','night stand':'cabinet','nightstand':'cabinet',
 'shelf':'shelf','shelves':'shelf','bookshelf':'shelf','book shelf':'shelf',
 'bed':'bed','chair':'chair','sofa':'sofa','couch':'sofa','bench':'bench','stool':'stool'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def canonical_label(text):
    text=re.sub(r'\s+',' ',str(text).lower().strip(' .'))
    return ALIASES.get(text.removeprefix('the '))

def frame_inventory():
    raw=json.loads((HERE/'grounded_surface_inventory.json').read_text())['scenes']
    lock=json.loads(LOCK.read_text());expected=set(lock['train_scenes'])
    assert len(expected)==563
    assert not expected & set(lock['validation_scenes']+lock['test_scenes'])
    available=set(raw)&expected
    return raw,lock,sorted(available),sorted(expected-available)

def run_detection(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    sys.path.insert(0,str(MODEL_ROOT/'sam2_official'))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sys.path.insert(0,'/path/to/research100/strict_real30_image_modules_20261102_v1')
    from image_reference_projection_v1 import project_instances
    torch.set_num_threads(2)
    raw,lock,scenes,missing=frame_inventory()
    queue=[]
    for scene in scenes:
        for rgb in raw[scene]['frames']:
            queue.append((scene,Path(rgb)))
    queue=[j for i,j in enumerate(queue) if i%args.shards==args.shard]
    if args.limit:queue=queue[:args.limit]
    out=args.output/'detections';out.mkdir(parents=True,exist_ok=True)
    todo=[];cached=0
    for scene,rgb in queue:
        key=scene+'_'+rgb.stem;record=out/(key+'.json')
        if record.exists():
            prior=json.loads(record.read_text())
            if prior.get('status') in ('detected','no_detection') and prior.get('prompt')==PROMPT and (out/(key+'.npz')).exists():
                cached+=1;continue
        todo.append((scene,rgb))
    dp=AutoProcessor.from_pretrained(MODEL_ROOT/'dino_base',local_files_only=True)
    dm=AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ROOT/'dino_base',local_files_only=True).to('cuda').eval()
    sp=SAM2ImagePredictor(build_sam2('configs/sam2/sam2_hiera_s.yaml',str(MODEL_ROOT/'sam2_original_checkpoint.pt'),device='cuda',mode='eval'))
    models={'dino_config_sha256':sha(MODEL_ROOT/'dino_base/config.json'),
            'sam2_checkpoint_sha256':sha(MODEL_ROOT/'sam2_original_checkpoint.pt'),
            'dino':'frozen_grounding_dino_base','sam2':'frozen_official_sam2_hiera_small'}
    write(args.output/f'detection_protocol_rank{args.shard}.json',dict(source_sha256=sha(__file__),models=models,
        prompt=PROMPT,split_lock_sha256=sha(LOCK),scenes=len(scenes),missing_frame_scenes=missing,
        no_GT_labels=True,box_threshold=.3,text_threshold=.25,projection='existing_RGBD_intrinsics_identity_color_depth_extrinsic_approximation',
        frame_rule='all_available_preregistered_RGBD_frames',shards=args.shards,shard=args.shard))
    started=time.monotonic();counts=Counter()
    for index,(scene,rgbfile) in enumerate(todo):
        tick=time.monotonic();folder=rgbfile.parent.parent;key=scene+'_'+rgbfile.stem
        row=dict(scene=scene,frame=rgbfile.stem,prompt=PROMPT,source='grounding_dino_official_sam2',
                 calibration='identity_color_depth_extrinsic_approximation',rgb_sha256=sha(rgbfile))
        try:
            rgb=np.asarray(Image.open(rgbfile).convert('RGB')).copy();im=Image.fromarray(rgb)
            depth=np.asarray(Image.open(folder/'depth'/(rgbfile.stem+'.png')),float)*.001
            pose=np.loadtxt(folder/'pose'/(rgbfile.stem+'.txt'))
            kd=np.loadtxt(folder/'intrinsics_depth.txt')[:3,:3];kc=np.loadtxt(folder/'intrinsics_color.txt')[:3,:3]
            x=dp(images=im,text=PROMPT,return_tensors='pt').to('cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):y=dm(**x)
            kw=dict(text_threshold=.25,target_sizes=[im.size[::-1]])
            kw['threshold' if 'threshold' in inspect.signature(dp.post_process_grounded_object_detection).parameters else 'box_threshold']=.3
            d=dp.post_process_grounded_object_detection(y,x.input_ids,**kw)[0]
            boxes=d['boxes'].float().cpu().numpy();scores=d['scores'].float().cpu().numpy()
            labels=list(d.get('text_labels',d.get('labels',[])))
            chosen=[i for i,label in enumerate(labels) if canonical_label(label) is not None]
            boxes=boxes[chosen];scores=scores[chosen];labels=[canonical_label(labels[i]) for i in chosen]
            masks=[]
            if len(boxes):
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    sp.set_image(rgb)
                    # Shared image embedding; all detected boxes admitted in bounded batches.
                    for first in range(0,len(boxes),8):
                        mask,_,_=sp.predict(box=boxes[first:first+8],multimask_output=False)
                        masks.extend(np.asarray(mask).reshape(-1,im.height,im.width).astype(bool))
            masks=np.asarray(masks,bool).reshape(-1,im.height,im.width)
            dets=project_instances(masks,depth,kd,kc,pose,color_from_depth=None,calibration_verified=False)
            packed={'boxes':boxes,'scores':scores,'label_ids':np.asarray([LABELS.index(x) for x in labels],np.int16)}
            # Projection retains all selected depth observations, without semantic instance metadata.
            for i,det in enumerate(dets):packed[f'points_world_{i}']=np.asarray(det['points_world'],np.float32)
            np.savez_compressed(out/(key+'.npz'),**packed)
            row.update(status='detected' if len(boxes) else 'no_detection',detections=len(boxes),labels=labels,
                       observed_points=[len(q['points_world']) for q in dets],npz_sha256=sha(out/(key+'.npz')))
        except Exception as e:
            row.update(status='failed',error=type(e).__name__+': '+str(e)[:500])
        row['seconds']=time.monotonic()-tick;write(out/(key+'.json'),row);counts[row['status']]+=1
        progress=dict(completed=index+1,total=len(todo),cached=cached,counts=dict(counts),seconds=time.monotonic()-started,shard=args.shard)
        write(args.output/f'detection_progress_rank{args.shard}.json',progress)
        print(json.dumps(progress),flush=True)
    write(args.output/f'detection_done_rank{args.shard}.json',dict(status='complete' if not counts['failed'] else 'with_failures',counts=dict(counts),cached=cached,new=len(todo)))

def old_source_index(scenes):
    records={s:[] for s in scenes}
    for path in sorted(OLD_CACHE.glob('*.json')):
        if len(path.stem)!=24:continue
        row=json.loads(path.read_text())
        if row.get('scene') not in records or row.get('status')!='detected':continue
        label=canonical_label(row.get('detector_phrase',''))
        if label is not None and path.with_suffix('.npz').exists():
            records[row['scene']].append(dict(json=str(path),npz=str(path.with_suffix('.npz')),label=label,frame=row['frame'],old=True))
    return records

def build_scene(job):
    from scipy.spatial import cKDTree
    scene,sources,entry,outdir,source_type=job;outdir=Path(outdir);tick=time.monotonic()
    assert sha(entry['path'])==entry['sha256']
    with np.load(entry['path'],allow_pickle=False) as z:world=np.asarray(z['vertices'],float)
    tree=cKDTree(world)
    point_scores={};frames_by_key={};source_records=[];raw_total=accepted_total=0
    for record in sources:
        meta=json.loads(Path(record['json']).read_text());npz_path=Path(record['npz'])
        with np.load(npz_path,allow_pickle=False) as z:
            scores=np.asarray(z['scores'],float)
            for i,score in enumerate(scores):
                key=f'projection_{i}_points_world' if record.get('old') else f'points_world_{i}'
                if key not in z:continue
                xyz=np.asarray(z[key],float).reshape(-1,3)
                xyz=xyz[np.isfinite(xyz).all(1)]
                # Uniform deterministic spread caps query load, not admitted object count.
                if len(xyz)>24000:xyz=xyz[np.linspace(0,len(xyz)-1,24000,dtype=int)]
                if not len(xyz):continue
                dist,ids=tree.query(xyz,k=1,workers=1);valid=dist<=.04
                raw_total+=len(xyz);accepted_total+=int(valid.sum())
                label=record['label'] if record.get('old') else LABELS[int(z['label_ids'][i])]
                label_id=LABELS.index(label);confidence=float(score)*float(valid.mean())
                for pid in np.unique(ids[valid]):
                    token=int(pid),label_id
                    point_scores[token]=max(point_scores.get(token,0.),confidence)
                    frames_by_key.setdefault(token,set()).add(record['frame'])
        source_records.append(dict(json=record['json'],npz_sha256=sha(npz_path),frame=record['frame']))
    keys=sorted(point_scores)
    arrays=dict(point_ids=np.asarray([x[0] for x in keys],np.int64),label_ids=np.asarray([x[1] for x in keys],np.int16),
                confidence=np.asarray([point_scores[x] for x in keys],np.float32),
                observation_counts=np.asarray([len(frames_by_key[x]) for x in keys],np.int16))
    outdir.mkdir(parents=True,exist_ok=True);np.savez_compressed(outdir/(scene+'.npz'),**arrays)
    report=dict(scene=scene,labels=LABELS,source='grounding_dino_sam2',source_mode=source_type,
        coordinate_frame='original_scene_vertex_ids',mesh_sha256=entry['sha256'],mesh_vertices=len(world),
        point_label_records=len(keys),unique_vertices=len({x[0] for x in keys}),frames=len({r['frame'] for r in sources}),
        projected_samples=raw_total,matched_samples=accepted_total,nearest_surface_gate_m=.04,
        confidence='DINO score multiplied by projection-to-mesh agreement fraction; uncalibrated soft evidence',
        color_depth_extrinsics='identity_approximation; not claimed calibrated',
        no_ground_truth_semantic_or_instance_labels=True,sources=source_records,seconds=time.monotonic()-tick,
        npz_sha256=sha(outdir/(scene+'.npz')))
    write(outdir/(scene+'.json'),report)
    return {k:v for k,v in report.items() if k not in ('sources','labels')}

def run_association(args):
    raw,lock,scenes,missing=frame_inventory();manifest=json.loads(MESH.read_text())
    if args.limit:scenes=scenes[:args.limit]
    records=old_source_index(scenes) if args.reuse_old else {s:[] for s in scenes}
    if not args.reuse_old:
        for path in sorted((args.output/'detections').glob('scene*.json')):
            row=json.loads(path.read_text())
            if row.get('scene') in records and row.get('status')=='detected':
                records[row['scene']].append(dict(json=str(path),npz=str(path.with_suffix('.npz')),frame=row['frame'],old=False))
    started=time.monotonic();results=[];outdir=args.output/('grounded_surfaces_cached' if args.reuse_old else 'grounded_surfaces')
    jobs=[(s,records[s],manifest['scenes'][s],str(outdir),'existing_predicted_masks' if args.reuse_old else 'fixed_support_prompt_full_frames') for s in scenes]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(build_scene,j) for j in jobs]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            progress=dict(completed=len(results),total=len(jobs),seconds=time.monotonic()-started,
                          scenes_with_surface_evidence=sum(r['unique_vertices']>0 for r in results),last_scene=result['scene'])
            write(args.output/('association_cached_progress.json' if args.reuse_old else 'association_progress.json'),progress)
            print(json.dumps(progress),flush=True)
    write(outdir/'index.json',dict(status='complete',scenes=len(results),expected_train_scenes=563,missing_frame_scenes=missing,
        split_lock_sha256=sha(LOCK),source_manifest_sha256=sha(MESH),labels=LABELS,
        coordinate_frame='original_scene_vertex_ids',source='grounding_dino_sam2',
        scenes_with_surface_evidence=sum(r['unique_vertices']>0 for r in results),scenes_without_surface_evidence=[r['scene'] for r in results if not r['unique_vertices']],
        no_GT_labels=True,records=results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['detect','associate'],required=True)
    p.add_argument('--output',type=Path,default=HERE/'grounded_support')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--reuse-old',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    run_detection(a) if a.mode=='detect' else run_association(a)
