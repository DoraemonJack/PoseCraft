import importlib.util,json,time,collections,pathlib,traceback
R=pathlib.Path('/path/to/research300/refineplace_v1_epoch2_fulltest_20261103_v1');O=pathlib.Path('/path/to/research300/posecraft_terra_parse_epoch8_20261116');D=O/'detector';status=O/'association_status.json'
def save(x):status.write_text(json.dumps(x,indent=2))
try:
 jobs=[json.loads(x) for x in open(O/'missing_jobs.jsonl')];expected=len(jobs)*4
 while len(list(D.glob('*.json')))<expected:
  n=len(list(D.glob('*.json')));save(dict(stage='waiting_detector',completed=n,expected=expected));time.sleep(10)
 groups=collections.defaultdict(list);errors=[]
 for p in D.glob('*.json'):
  x=json.loads(p.read_text());groups[x['scene'],x['detector_phrase']].append((x,p))
  if x['status']=='failed':errors.append(x)
 assert len(groups)==len(jobs) and not errors,(len(groups),len(jobs),errors[:5])
 save(dict(stage='associating',groups=len(groups)))
 spec=importlib.util.spec_from_file_location('catalog_builder',R/'build_test_reference_catalog.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=D;m.torch.set_num_threads(2)
 features,counts=m.associate(groups,m.read(R/'geometry/mesh_manifest_v6.json'));assert not counts.get('features_failed'),counts
 save(dict(stage='complete',groups=len(groups),counts=counts,nonempty_groups=sum(bool(v) for v in features.values())))
except BaseException as e:save(dict(stage='failed',error=repr(e),traceback=traceback.format_exc()));raise
