import pathlib,json,importlib.util,collections,traceback
R=pathlib.Path("/path/to/research300/posecraft_val_reference_correction_20261117"); base=pathlib.Path("/path/to/research300/refineplace_v1_epoch2_fulltest_20261103_v1"); D=R/"detector"
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
status=R/"association_status.json"
try:
 g=json.loads((base/"geometry/mesh_manifest_v6.json").read_text()); tasks=[json.loads(x) for x in (R/"validation_prompts.jsonl").read_text().splitlines()]; scenes={x["scene"] for x in tasks}
 builder=load(base/"build_test_geometry.py","geometry_builder"); output=R/"geometry/scenes";output.mkdir(parents=True,exist_ok=True)
 for scene in sorted(scenes-set(g["scenes"])):
  ply=pathlib.Path("/path/to/research300/refineplace_v22_20261106_v1/val_scans_repaired_v1")/scene/(scene+"_vh_clean_2.ply"); txt=R/"scene_metadata"/(scene+".txt")
  vertices,faces,raw_rgb,xyz,rgb,mean=builder.read_raw_ply(ply); path=output/(scene+".npz");builder.np.savez_compressed(path,vertices=vertices,faces=faces,raw_rgb=raw_rgb,xyz=xyz,rgb=rgb)
  g["alignments"][scene]=builder.alignment(txt).tolist();g["scenes"][scene]=dict(scene_id=scene,path=str(path),sha256=builder.sha(path),source_ply=str(ply),source_txt=str(txt),observation_mean_translation_m=mean.astype(float).tolist(),observation_path=str(path),observation_xyz_field="xyz",observation_rgb_field="rgb")
 (R/"geometry/val_mesh_manifest.json").write_text(json.dumps(g,indent=2)); print("geometry ready",len(scenes),flush=True)
 grouped=collections.defaultdict(list)
 for p in D.glob("*.json"):
  x=json.loads(p.read_text()); assert x["status"]!="failed",x;grouped[x["scene"],x["detector_phrase"]].append((x,p))
 jobs=[json.loads(x) for x in (R/"new_jobs.jsonl").read_text().splitlines()]; assert set(grouped)=={(x["scene"],x["detector_phrase"]) for x in jobs},(len(grouped),len(jobs))
 m=load(base/"build_test_reference_catalog.py","catalog_builder_val");m.OUT=D;m.FRAMES=R/"frames";m.torch.set_num_threads(2);features,counts=m.associate(grouped,g);assert not counts.get("features_failed"),counts
 result=dict(status="complete",groups=len(grouped),counts=counts,nonempty_groups=sum(bool(v) for v in features.values()));status.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except BaseException as e:
 status.write_text(json.dumps(dict(status="failed",error=repr(e),traceback=traceback.format_exc()),indent=2));raise
