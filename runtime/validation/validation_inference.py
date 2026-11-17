import argparse,importlib.util,json,os,sys,time
from pathlib import Path
R=Path("/path/to/research300/posecraft_val_reference_correction_20261117");SOURCE=Path("/path/to/research300/refineplace_orientation_trust_v19_20261103_v1");DATA=Path("/path/to/research300/refineplace_v1_epoch2_fulltest_20261103_v1");TRAIN=Path("/path/to/research300/refineplace_v2_main_retrain_20261103_v1")
sys.path.insert(0,"/path/to/research300/posecraft_terra_parse_epoch8_20261116");from negative_distance import wrap_negative
p=argparse.ArgumentParser();p.add_argument("--steps",type=int,required=True);p.add_argument("--offset",type=int,required=True);p.add_argument("--limit",type=int,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--resume",action="store_true");a=p.parse_args();assert a.steps in range(2,27,2) and a.offset>=0 and a.offset+a.limit<=2566
source=(SOURCE/"infer_retrained_pair_fulltest.py").read_text(); source=source.replace("3500","2566").replace("manifest['scene_count'] == 134","manifest['scene_count'] == 141").replace("manifest['split'] == 'test'", "manifest['split'] == 'val'")
source=source.replace("len(saved['path']) > 2","len(saved['path']) > MAX_ACTION_STEPS").replace("len(refinement) > 2","len(refinement) > 8")
source=source.replace("assert {t['scene'] for t in all_tasks} == set(lock['test_scenes'])", "assert len({t['scene'] for t in all_tasks}) == 141 and not {t['scene'] for t in all_tasks} & set(lock['train_scenes'])")
source=source.replace("assert grounded['status'] == 'complete' and {r['scene'] for r in grounded['records']} == set(lock['test_scenes'])", "assert grounded['status'] == 'complete' and {r['scene'] for r in grounded['records']} == {t['scene'] for t in all_tasks}")
source=source.replace("row = dict(task, split='test'", "row = dict(task, split='val'").replace("q['split'] == 'test'", "q['split'] == 'val'")
source=source.replace("for task in pending:\n            i, uid", "for task in pending:\n            task_start = time.monotonic()\n            i, uid")
source=source.replace("record = d.clean(record)", "record['inference_seconds'] = time.monotonic() - task_start\n            record = d.clean(record)")
key="            scorer, pair_info = scorer_factory.for_task(task['prompt'], refs[(task['scene'], task['group_id'])], inferred)";assert key in source;source=source.replace(key,key+"\n            scorer = wrap_negative(scorer, refs[(task['scene'], task['group_id'])], row['centers'], ctx['candidate_mask'])")
spec=importlib.util.spec_from_file_location("val_inference",SOURCE/"infer_retrained_pair_fulltest.py");runtime=importlib.util.module_from_spec(spec);runtime.__dict__.update(MAX_ACTION_STEPS=a.steps,wrap_negative=wrap_negative);exec(compile(source,str(SOURCE/"infer_retrained_pair_fulltest.py"),"exec"),runtime.__dict__)
oldload=runtime.load_runtime
def load(root):
 m=oldload(root)
 sys.path.insert(0,str(SOURCE))
 original=m.load_mesh_manifest
 def val_mesh(path):
  from test_asset_library import TestAssetLibrary,sha
  class ValLibrary(TestAssetLibrary):
   def __init__(self,path):
    self.path=Path(path);self.manifest=json.loads(self.path.read_text());self.manifest_sha256=sha(self.path); mm=self.manifest;ts=json.loads((R/"tasks_manifest.json").read_text())["tasks"]; scenes={t["scene"] for t in ts};assets={t["asset_uuid"] for t in ts}
    assert scenes<=set(mm["scenes"]) and assets<=set(mm["assets"]) and scenes<=set(mm["test_scenes"]);self.geometry_config=dict(mm["geometry_config"]);self.assets={};self._verified=set();self.scene_alignment_audits={}
  return ValLibrary(path)
 m.load_mesh_manifest=val_mesh
 import refiner_v19
 s=(SOURCE/"refiner_v19.py").read_text();assert "0 <= max_steps <= 2" in s;s=s.replace("0 <= max_steps <= 2","0 <= max_steps <= 8");exec(compile(s,str(SOURCE/"refiner_v19.py"),"exec"),refiner_v19.__dict__)
 rollout=m.rollout.rollout_batch;refine=refiner_v19.refine_after_baseline_v19
 def roll(*args,**kwargs):kwargs["steps"]=a.steps;return rollout(*args,**kwargs)
 def rot(*args,**kwargs):kwargs["max_steps"]=8;return refine(*args,**kwargs)
 m.rollout.rollout_batch=roll;refiner_v19.refine_after_baseline_v19=rot
 return m
runtime.load_runtime=load
oldwrite=runtime.write
def write(path,value):
 if Path(path).name in ("protocol_before_forward.json","summary.json"):
  value.update(split="val",run_kind="validation_inference",steps=a.steps,baseline_rollout_steps=a.steps,refinement_max_steps=8,scope="validation split",timing_scope="per-task candidate preparation plus policy and geometry inference; excludes scene setup and grounding preprocessing")
 oldwrite(path,value)
runtime.write=write
args=argparse.Namespace(checkpoint=TRAIN/"rank0/epoch8.pt",training_protocol=TRAIN/"protocol_before_training.json",output=a.out,pair_head=None,data_root=R,references=R/"catalog/reference_catalog_compact_v4_test.jsonl",references_summary=R/"catalog/manifest.json",grounded_root=R/"support/grounded_surfaces",baseline_predictions=None,v11_baseline_predictions=None,limit=a.limit,offset=a.offset,resume=a.resume,preflight_only=False)
runtime.main(args)
