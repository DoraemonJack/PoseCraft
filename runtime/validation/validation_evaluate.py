import sys,os,importlib.util,pathlib,json,numpy as np
E=pathlib.Path("/path/to/home/placeit3d_near10_run");R=pathlib.Path("/path/to/research300/posecraft_val_reference_correction_20261117");sys.path.insert(0,str(E));spec=importlib.util.spec_from_file_location("val_near10_trial",E/"run_trial.py");m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
m.ASSETS=pathlib.Path("/path/to/research300/placeit_mesh_v6/assets");m.H5=pathlib.Path("/path/to/home/codex_3d_research_20261028/placeit3d_reproduction_v1/downloads/essential/placeit3d_val.h5");m.SCANS=R/"evaluation_scans";m.OUT=R/"step_experiment/evaluation";m.OUT.mkdir(parents=True,exist_ok=True);m.WORKERS=int(os.environ.get("WORKERS","6"));m.NEAR_FRACTION=0.10;m.constraints.FACING_ANGLE_RAD=np.deg2rad(20.0);m.MESHES={};m.ENTRIES={e.index:e for e in m.load_entries(m.H5)};assert len(m.ENTRIES)==2566
if len(sys.argv)>1 and sys.argv[1]=="smoke":
 rows=json.loads((R/"smoke_val3/predictions.json").read_text());groups={}
 for row in rows:groups.setdefault(m.ENTRIES[row["entry_id"]].scene_id,[]).append(row)
 for scene,v in groups.items():
  result=m._scene_job((scene,v,str(m.SCANS)));print(json.dumps(result),flush=True);assert not result["error"] and len(result["rows"])==len(v)
else:
 name=sys.argv[1];p=pathlib.Path(sys.argv[2]);original=m.aggregate
 def collect(placements):
  (m.OUT/(name+"_per_task.json")).write_text(json.dumps(placements))
  if len(placements)!=2566:raise RuntimeError("Incomplete evaluation: "+str(len(placements)))
  errors=[x for x in placements if x.get("errors")];(m.OUT/(name+"_constraint_errors.json")).write_text(json.dumps(errors))
  return original(placements)
 m.aggregate=collect;m.evaluate(name,p)
 failures=json.loads((m.OUT/(name+"_failures.json")).read_text())
 if failures:raise RuntimeError("Scene evaluation failures: "+str(len(failures)))

