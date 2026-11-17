"""Official TEST observation-only manifest and unchanged frozen text encoder."""
import hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import h5py,numpy as np,torch

ROOT=Path(__file__).resolve().parent
LOCK=Path('/path/to/research100/official_split_training_audit_20261102_v1/official563_training_split_lock.json')
MODELROOT=Path('/path/to/research300/refineplace_v1_0_20261103_v1')
BASE=Path('/path/to/home/codex_3d_research_20261028')
sys.path.insert(0,'/path/to/research100/native_val297_current_model_20261102_v1')
from run_native_val297_allmask_v1 import Encoder,model_identity

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as fp:
        for b in iter(lambda:fp.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)

def main():
    torch.set_num_threads(2);ROOT.mkdir(parents=True,exist_ok=True)
    lock=json.loads(LOCK.read_text());source=Path(lock['official_h5']['test']['source'])
    assert sha(source)==lock['official_h5']['test']['sha256']
    tasks=[];paths=[]
    with h5py.File(source,'r') as h:
        for name,g in h.items():
            if name=='metadata' or not isinstance(g,h5py.Group):continue
            scene=name[:12];assert scene in lock['test_scenes'] and scene not in lock['train_scenes']
            dims=np.asarray(g['asset_size'][()],float);assert dims.shape==(3,) and np.isfinite(dims).all() and (dims>0).all()
            for entry in g:
                if g.get(entry,getclass=True) is not h5py.Group or len(g[entry])==0:continue
                prompt=g[entry]['prompt'][()];prompt=prompt.decode() if isinstance(prompt,bytes) else str(prompt)
                i=len(tasks);paths.append(name+'/'+entry)
                tasks.append(dict(original_entry_id=i,group_id=i,scene=scene,asset_scene=name,entry=entry,
                                  asset_uuid=name.split('_')[-1],asset_size=dims.tolist(),prompt=prompt,
                                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
    assert len(tasks)==3500 and len({t['scene'] for t in tasks})==134 and len({t['asset_uuid'] for t in tasks})==20
    order=hashlib.sha256(json.dumps(paths,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert order=='d264ce8de42591de7565d7ec13008f1f9e5641d208890261aa97a637db9a7464'
    manifest=dict(split='test',tasks=tasks,task_count=len(tasks),scene_count=134,asset_count=20,
                  h5_source=str(source),h5_sha256=sha(source),h5_entry_paths_sha256=order,
                  split_lock_sha256=sha(LOCK),h5_value_fields_read=['prompt','asset_size'],
                  placement_labels_read=False,ground_truth_relationships_read=False,ground_truth_instances_used=False)
    p=ROOT/'tasks_manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest
    else:write(p,manifest)
    rp=BASE/'fullscene_upgrade_v2/roberta-base'
    original=json.loads((BASE/'rl_position_adjustment_v1/joint_text_cache_v2/manifest.json').read_text())
    identity=model_identity(rp);assert identity['signature']==original['model_signature'] and original['max_length']==128
    dest=ROOT/'text_features.npy';meta=ROOT/'text_features.json'
    if dest.exists():
        old=json.loads(meta.read_text());assert old['manifest_sha256']==sha(p) and old['array_sha256']==sha(dest);return
    started=time.monotonic()
    encoder=Encoder(SimpleNamespace(model=rp,device='cuda:0',batch_size=16,max_length=128))
    values=encoder.encode([t['prompt'] for t in tasks]);assert values.shape==(3500,768) and np.isfinite(values).all()
    np.save(dest,values)
    write(meta,dict(status='complete',manifest_sha256=sha(p),array_sha256=sha(dest),shape=list(values.shape),
                    encoder=identity,max_length=128,weights_updated=False,seconds=time.monotonic()-started,
                    placement_labels_read=False,ground_truth_relationships_read=False))
    print(json.dumps(dict(status='complete',tasks=3500,seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
