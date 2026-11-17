import json,pathlib,hashlib,sys,time
from types import SimpleNamespace
import numpy as np,h5py,torch
R=pathlib.Path("/path/to/research300/posecraft_val_reference_correction_20261117");BASE=pathlib.Path("/path/to/home/codex_3d_research_20261028");H=BASE/"placeit3d_reproduction_v1/downloads/essential/placeit3d_val.h5"; LOCK=pathlib.Path("/path/to/research100/official_split_training_audit_20261102_v1/official563_training_split_lock.json")
sys.path.insert(0,"/path/to/research100/native_val297_current_model_20261102_v1");from run_native_val297_allmask_v1 import Encoder,model_identity
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
rows=[json.loads(x) for x in (R/"mapped_val_tasks.jsonl").read_text().splitlines()];tasks=[]; paths=[]
with h5py.File(H,"r") as h:
 for r in rows:
  i=r["entry_id"];g=h[r["asset_scene"]];dim=np.asarray(g["asset_size"][()],float);assert dim.shape==(3,) and np.isfinite(dim).all() and (dim>0).all()
  p=g[r["entry"]]["prompt"][()];p=p.decode() if isinstance(p,bytes) else str(p);assert p==r["prompt"]
  t=dict(original_entry_id=i,group_id=i,scene=r["scene"],asset_scene=r["asset_scene"],entry=r["entry"],asset_uuid=r["asset_scene"].split("_")[-1],asset_size=dim.tolist(),prompt=p,prompt_sha256=hashlib.sha256(p.encode()).hexdigest());tasks.append(t);paths.append(t["asset_scene"]+"/"+t["entry"])
  r.update(t)
assert len(tasks)==2566 and [t["original_entry_id"] for t in tasks]==list(range(len(tasks)))
(R/"mapped_val_tasks.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
manifest=dict(split="val",tasks=tasks,task_count=len(tasks),scene_count=len({x["scene"] for x in tasks}),asset_count=len({x["asset_uuid"] for x in tasks}),h5_source=str(H),h5_sha256=sha(H),h5_entry_paths_sha256=hashlib.sha256(json.dumps(paths,separators=(",",":"),ensure_ascii=False).encode()).hexdigest(),split_lock_sha256=sha(LOCK),h5_value_fields_read=["prompt","asset_size"],placement_labels_read=False,ground_truth_relationships_read=False,ground_truth_instances_used=False)
p=R/"tasks_manifest.json";p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
rp=BASE/"fullscene_upgrade_v2/roberta-base";orig=json.loads((BASE/"rl_position_adjustment_v1/joint_text_cache_v2/manifest.json").read_text());identity=model_identity(rp);assert identity["signature"]==orig["model_signature"] and orig["max_length"]==128
torch.set_num_threads(2);start=time.monotonic();encoder=Encoder(SimpleNamespace(model=rp,device="cuda:0",batch_size=64,max_length=128));values=encoder.encode([t["prompt"] for t in tasks]);assert values.shape==(len(tasks),768) and np.isfinite(values).all();np.save(R/"text_features.npy",values)
(R/"text_features.json").write_text(json.dumps(dict(status="complete",manifest_sha256=sha(p),array_sha256=sha(R/"text_features.npy"),shape=list(values.shape),encoder=identity,max_length=128,seconds=time.monotonic()-start),indent=2));print(json.dumps(dict(status="complete",tasks=len(tasks),seconds=time.monotonic()-start)),flush=True)
