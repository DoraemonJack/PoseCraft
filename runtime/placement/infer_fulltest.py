"""One frozen epoch2 forward pass over all official TEST instructions.

Reads observations and predicted reference/receiver inputs. Does not open
placement labels, relationship labels, segmentation types or instance targets.
"""
import argparse,gc,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
V1=Path('/path/to/research300/refineplace_v1_0_20261103_v1')
MODULES=Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
OLD=Path('/path/to/research300/refineplace_rl_geometry_10k_20261102_v1/joint_collision_orientation')
sys.path[:0]=[str(ROOT),str(V1),str(MODULES),str(OLD),'/path/to/research100/native_val297_current_model_20261102_v1']
import train_official563_joint_v7_sync_full as d
from model_v10 import model_from_checkpoint
import joint_rollout_v1 as rollout
from test_asset_library import load_mesh_manifest
from wrapped_geometry_backend_v10 import install_geometry
from support_regions import prepare_scene,prepare_asset,load_grounded_surfaces,FEATURE_NAMES
from v1_scene import build_task_candidates
from run_native_val297_allmask_v1 import candidate_ids,raw_features

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fp:
        for b in iter(lambda:fp.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def write(path,data):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d.clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)

def main(a):
    torch.set_num_threads(2);torch.cuda.set_device(0)
    os.environ['REFINEPLACE_WRAPPED_FIELDS']=str(ROOT/'wrapped_fields/manifest.json')
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=read(ROOT/'tasks_manifest.json');tasks=manifest['tasks']
    assert manifest['split']=='test' and len(tasks)==3500 and manifest['scene_count']==134
    lock=read(d.LOCK);assert manifest['split_lock_sha256']==sha(d.LOCK)
    assert {t['scene'] for t in tasks}==set(lock['test_scenes']) and not set(lock['test_scenes'])&set(lock['train_scenes'])
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert cp['epoch']==2 and cp['architecture']=='RefinePlaceV10'
    assert cp['normalization_sha256']==sha(d.F/'normalization.json') and cp['official_split_lock_sha256']==sha(d.LOCK)
    assert cp['protocol_sha256']==sha(a.training_protocol)
    textmeta=read(ROOT/'text_features.json');assert textmeta['manifest_sha256']==sha(ROOT/'tasks_manifest.json')
    assert textmeta['array_sha256']==sha(ROOT/'text_features.npy')
    texts=np.load(ROOT/'text_features.npy',mmap_mode='r',allow_pickle=False);assert texts.shape==(3500,768)
    refmeta=read(a.references_summary);assert refmeta['status']=='complete'
    assert refmeta['data_sha256']==sha(a.references)
    assert refmeta['task_manifest_sha256']==sha(ROOT/'tasks_manifest.json')
    refs={}
    for line in a.references.read_text(encoding='utf-8').splitlines():
        r=json.loads(line);key=r['scene'],r['group_id'];assert key not in refs;refs[key]=r
    assert len(refs)==3500 and {(t['scene'],t['group_id']) for t in tasks}==set(refs)
    for task in tasks:
        r=refs[task['scene'],task['group_id']]
        assert all(r[k]==task[k] for k in ('scene','group_id','asset_scene','entry','prompt_sha256'))
    grounded=read(a.grounded_root/'index.json')
    assert grounded['status']=='complete' and {r['scene'] for r in grounded['records']}==set(lock['test_scenes'])
    assert grounded['split_lock_sha256']==sha(d.LOCK)
    grounded_audit=read(a.grounded_root.parent/'admission_audit.json')
    assert grounded_audit['status']=='passed' and grounded_audit['data_index_sha256']==sha(a.grounded_root/'index.json')
    library=load_mesh_manifest(ROOT/'geometry/mesh_manifest_v6.json')
    geometry_audit=read(ROOT/'geometry/geometry_admission_audit.json')
    assert geometry_audit['status']=='passed' and geometry_audit['manifest_sha256']==library.manifest_sha256
    norm=read(d.F/'normalization.json');mean,std=np.asarray(norm['mean']),np.asarray(norm['std'])
    types=read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types']
    model=model_from_checkpoint(d,types,len(FEATURE_NAMES),cp).cuda().eval().requires_grad_(False);del cp
    before=d.tiny.digest(model.state_dict());install_geometry()
    protocol=dict(split='test',tasks=3500,scenes=134,checkpoint_epoch=2,checkpoint_sha256=sha(a.checkpoint),
                  manifest_sha256=sha(ROOT/'tasks_manifest.json'),h5_sha256=manifest['h5_sha256'],
                  h5_entry_paths_sha256=manifest['h5_entry_paths_sha256'],normalization_sha256=sha(d.F/'normalization.json'),
                  reference_catalog_sha256=sha(a.references),reference_summary_sha256=sha(a.references_summary),
                  receiver_index_sha256=sha(a.grounded_root/'index.json'),geometry_manifest_sha256=library.manifest_sha256,
                  wrapped_fields_sha256=sha(ROOT/'wrapped_fields/manifest.json'),source_sha256=sha(__file__),
                  learned_policy_sha256=sha(V1/'model_v10.py'),region_code_sha256=sha(V1/'support_regions.py'),
                  deterministic=True,steps=2,test_labels_read=False,ground_truth_relationships_read=False,
                  instance_annotation_inputs=False,scope='Official complete TEST, previous project test exposure documented; not first-ever blind evaluation')
    sealed=a.output/'protocol_before_forward.json'
    if sealed.exists():assert read(sealed)==protocol,'Resume changed inputs or frozen checkpoint'
    else:write(sealed,protocol)
    done={}
    journal=a.output/'predictions_in_progress.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            r=json.loads(line);i=r['original_entry_id'];assert i not in done;done[i]=r
    start=time.monotonic();assets={};active=sum(int(r['reference_active']) for r in done.values())
    for scene in sorted(lock['test_scenes']):
        pending=[t for t in tasks if t['scene']==scene and t['original_entry_id'] not in done]
        if not pending:continue
        path,entry=library._file('scenes',scene)
        with np.load(path,allow_pickle=False) as z:
            xyz=z['xyz'].copy();rgb=z['rgb'].copy();world=z['vertices'].copy();faces=z['faces'].copy()
        assert xyz.shape==rgb.shape and xyz.shape==world.shape
        translation=np.asarray(entry['observation_mean_translation_m'],float)
        assert np.linalg.norm(world-xyz-translation,axis=1).max()<1e-5
        A=np.asarray(library.manifest['alignments'][scene],float)
        aligned=(xyz.astype(np.float64)@A[:3,:3].T).astype(np.float32)
        offset=A[:3,:3]@translation+A[:3,3]
        assert np.max(np.linalg.norm((world@A[:3,:3].T+A[:3,3])-(aligned+offset),axis=1))<2e-5
        g=load_grounded_surfaces(a.grounded_root,scene);assert g['metadata']['mesh_sha256']==entry['sha256']
        assert g['metadata']['coordinate_frame']=='original_scene_vertex_ids' and g['metadata']['mesh_vertices']==len(xyz)
        receiver=prepare_scene(aligned,scene,g,pool_size=8192)
        original=candidate_ids(xyz,scene);geometry_cache={};asset_priors={}
        adapter=d.paired.MeshSceneAdapterV6(aligned,faces,library)
        for task in pending:
            i=task['original_entry_id'];uid=task['asset_uuid'];dims=np.asarray(task['asset_size'],np.float32)
            if uid not in assets:
                assets[uid]=torch.load(d.b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/(uid+'.pt'),map_location='cpu',weights_only=True).reshape(-1).cuda()
            if uid not in asset_priors:
                vertices,afaces=library.asset(uid,dims);asset_priors[uid]=prepare_asset(receiver,dims,vertices,afaces)
            raw=raw_features(rgb,original,dims,task['prompt'])
            features=build_task_candidates(receiver,asset_priors[uid],original,task,(xyz,rgb,None,None,None),raw,A,mean,std,geometry_cache)
            row=dict(task,split='test',dims=dims,text=texts[i].copy(),**features)
            f,text,asset,ctx,decision,inferred=d.make_context([row],assets,refs,2,types)
            model.reference_head.inferred_mask=inferred
            rf=torch.as_tensor(row['region_features'][None],device='cuda',dtype=torch.float32)
            is_active=bool(ctx['reference_mask'].any() and (ctx['relation_mask']|inferred).any())
            query_count=0
            class Service:
                def query_many(self,queries):
                    nonlocal query_count
                    out=[]
                    for q in queries:
                        assert q['scene']==scene and q['split']=='test' and q['asset_uuid']==uid
                        adapter.set_asset(uid,dims);out.append(adapter.features(q['center'],q['dims'],q['yaw']));query_count+=1
                    return out
            obs=[{key:row[key] for key in ('scene','split','asset_uuid','dims','centers')}]
            with torch.inference_mode(),model.reference_batch(**ctx),model.region_batch(rf):
                tr=rollout.rollout_batch(model,f,text,asset,ctx['candidate_mask'],obs,Service(),steps=2,deterministic=True,
                                         apply_offsets=True,anti_cycle=True,mode_epsilon=0.)[0]
            position=np.asarray(tr['center'])+offset+np.array([0.,0.,float(dims[2])/2])
            result=dict(original_entry_id=i,scene=scene,asset_scene=task['asset_scene'],entry=task['entry'],asset_uuid=uid,
                        center=d.clean(tr['center']),yaw=float(tr['yaw']),reference_active=is_active,
                        region_report=row['region_report'],geometry_queries=query_count,
                        prediction=dict(entry_id=i,dataset_idx=-1,tx=float(position[0]),ty=float(position[1]),tz=float(position[2]),rotation=math.degrees(float(tr['yaw']))%360))
            with journal.open('a',encoding='utf-8') as fp:fp.write(json.dumps(d.clean(result),ensure_ascii=False)+'\n');fp.flush()
            done[i]=result;active+=is_active
            write(a.output/'progress.json',dict(status='inference',completed=len(done),total=3500,scene=scene,
                                               reference_active=active,elapsed_this_process=time.monotonic()-start,checkpoint_epoch=2))
            if len(done)%10==0:print(json.dumps(dict(completed=len(done),total=3500,scene=scene)),flush=True)
        del adapter,receiver,asset_priors,geometry_cache,world,faces,xyz,rgb,aligned
        gc.collect()
    assert set(done)==set(range(3500)) and before==d.tiny.digest(model.state_dict())
    output=a.output/'predictions.json';write(output,[done[i]['prediction'] for i in range(3500)])
    summary=dict(protocol,status='complete',scope='test',inference_complete=True,scene_count=134,weights_unchanged=True,
                 prediction_sha256=sha(output),reference_active=active,reference_masked=3500-active,
                 elapsed_this_process=time.monotonic()-start,protocol_sha256=sha(sealed))
    write(a.output/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('checkpoint','training-protocol','output','references','references-summary','grounded-root'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
