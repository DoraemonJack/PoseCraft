"""Explicit CPU8 dispatch adapter; no training entry point or automatic migration."""
import hashlib, importlib.util
from pathlib import Path
WORKERS=Path('/path/to/research100/refineplace_persistent_grouped_pool_modules_v2')
EXPECTED={'persistent_grouped_geometry_v2.py':'d10475ec23c497336e52cdfbb6bc4936789a80478821ea82f8d3a97de6417bb1','private_worker_geometry_v1.py':'4b214d2a0b2b6c41615cfd0aa0a0ecf773c81521a1997bac4d88240dc9b4eaee'}
def verify_sources():
 for name,expected in EXPECTED.items():
  actual=hashlib.sha256((WORKERS/name).read_bytes()).hexdigest()
  if actual!=expected:raise ValueError('CPU8 geometry source identity mismatch: '+name)
 return dict(EXPECTED)
class CPU8GeometryBackend:
 """Keep one instance for the entire trainer lifetime. Exceptions propagate."""
 def __init__(self):
  verify_sources()
  import sys
  sys.path.insert(0,str(WORKERS))
  from persistent_grouped_geometry_v2 import PersistentGroupedGeometryService
  self.service=PersistentGroupedGeometryService(workers=8,chunk_size=0)
  self.closed=False
 def query_many(self,tasks):
  if self.closed:raise RuntimeError('CPU8 backend is closed')
  # Worker wrapper validates all six observation fields before queueing;
  # grouping is within this call and its implementation restores input order.
  return self.service.query_many(tasks)
 def process_snapshot(self):return self.service.process_snapshot()
 def close(self):
  if not self.closed:self.service.close();self.closed=True
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
def observation_scene_class(driver):
 """Same admitted observation reader; omit unused parent geometry construction.
 Labels remain inherited/deferred until after rollout, exactly as v2.
 """
 class ObservationScene(driver.Scene):
  def __init__(self,s,lock,library):
   d=driver;d.guard(lock,s);self.s=s;self.lock=lock;self.path=d.F/s
   self.m=d.read(self.path/'manifest.json');assert self.m['partition']=='official_train'
   self.a={k:d.arr(self.path,self.m,k) for k in ['features','centers','candidate_ids','candidate_offsets','group_ids','text','dims','asset_uuid']}
   self.registry=d.read(self.path/'task_registry.json');assert d.sha(self.path/'task_registry.json')==self.m['task_registry_sha256']
   self.bygi={int(x):i for i,x in enumerate(self.a['group_ids'])};self.tm=d.read(d.TARGET/s/'manifest.json')
   assert d.sha(d.TARGET/s/'manifest.json')==self.m['source_identity']['observation_manifest']
   self.labels_arrays=None;self.label_cache={};self.trees={};self.adapter=None
 return ObservationScene
