"""Same frozen support detector/projection on fixed validation scene inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import build_grounded_surfaces as core

ROOT=Path(__file__).resolve().parent
MANIFEST=Path('/path/to/research300/refineplace_sync20_inferred_official150_20261102_v1/fixed_validation150.json')
FRAMES=ROOT/'validation_grounded_frames'

def frame_inventory():
    tasks=json.loads(MANIFEST.read_text())['tasks']
    expected=sorted({t['scene'] for t in tasks})
    lock=json.loads(core.LOCK.read_text())
    assert not set(expected)&set(lock['train_scenes'])
    assert set(expected)<=set(lock['validation_scenes'])
    raw={}
    for scene in expected:
        folder=FRAMES/'inputs'/scene
        frames=[str(p) for p in sorted((folder/'color').glob('*.jpg'))
                if (folder/'depth'/(p.stem+'.png')).exists() and (folder/'pose'/(p.stem+'.txt')).exists()]
        if frames:raw[scene]={'folder':str(folder),'frames':frames}
    assert len(raw)==len(expected)
    return raw,lock,expected,[]

def run_association(args):
    raw,lock,scenes,missing=frame_inventory()
    records={s:[] for s in scenes};manifest=json.loads(core.MESH.read_text())
    for p in sorted((args.output/'detections').glob('scene*.json')):
        row=json.loads(p.read_text())
        assert row['status']!='failed',(str(p),row)
        if row['scene'] in records and row['status']=='detected':
            records[row['scene']].append(dict(json=str(p),npz=str(p.with_suffix('.npz')),frame=row['frame'],old=False))
    out=args.output/'grounded_surfaces'
    jobs=[(s,records[s],manifest['scenes'][s],str(out),'fixed_support_prompt_full_frames') for s in scenes]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(core.build_scene,j) for j in jobs]):results.append(f.result())
    core.write(out/'index.json',dict(status='complete',partition='fixed_validation150',tasks=150,scenes=len(results),
        expected_validation_scenes=scenes,missing_frame_scenes=missing,split_lock_sha256=core.sha(core.LOCK),
        task_manifest_sha256=core.sha(MANIFEST),frame_identity_sha256=core.sha(FRAMES/'frame_source_identity_before_models.json'),
        source_manifest_sha256=core.sha(core.MESH),labels=core.LABELS,coordinate_frame='original_scene_vertex_ids',
        source='grounding_dino_sam2',scenes_with_surface_evidence=sum(r['unique_vertices']>0 for r in results),
        scenes_without_surface_evidence=[r['scene'] for r in results if not r['unique_vertices']],no_GT_labels=True,records=results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['detect','associate'],required=True)
    p.add_argument('--output',type=Path,default=ROOT/'validation_grounded_support')
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=10)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    core.frame_inventory=frame_inventory
    core.write(a.output/'validation_protocol.json',dict(partition='fixed_validation150',task_manifest_sha256=core.sha(MANIFEST),
        source_sha256=core.sha(__file__),shared_detector_source_sha256=core.sha(core.__file__),
        frame_identity_sha256=core.sha(FRAMES/'frame_source_identity_before_models.json'),no_GT_labels=True))
    core.run_detection(a) if a.mode=='detect' else run_association(a)
