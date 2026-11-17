"""Actual TRAIN observations only: wrapped FCL pool and support-preservation smoke."""
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
sys.path.insert(0,'/path/to/research100/refineplace_persistent_grouped_pool_modules_v2')
from wrapped_geometry_backend_v10 import CollisionGeometryBackend
import private_worker_geometry_v1 as private


def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x


def main():
    source=HERE.parent/'refineplace_rl_geometry_10k_20261102_v1/joint_collision_orientation/geometry_component_coverage.json'
    records=json.loads(source.read_text(encoding='utf-8'))['records']
    first=records[0];second=next(r for r in records if r['scene']==first['scene'] and r['asset_uuid']!=first['asset_uuid'])
    tasks=[{k:r[k] for k in ('scene','split','asset_uuid','dims','center','yaw')} for r in (first,second)]
    private.initialize()
    original=[]
    for task in tasks:
        value,_=private.query_private(task)
        original.append(value)
    started=time.monotonic()
    with CollisionGeometryBackend(workers=2) as backend:
        wrapped=backend.query_many(tasks)
        first_seconds=time.monotonic()-started
        tick=time.monotonic();repeat=backend.query_many(tasks);cached_seconds=time.monotonic()-tick
        processes=backend.process_snapshot()
    report=[]
    for task,a,b,again in zip(tasks,original,wrapped,repeat):
        av,ast,_=a;bv,bst,_=b
        for key in ('support_contact_fraction','support_known_fraction','support_unknown_fraction','support_gap_fraction',
                    'support_sample_count','support_samples_world','support_distances_m','support_grid_side'):
            np.testing.assert_array_equal(ast[key],bst[key],err_msg=key)
        assert len(bv)==26 and len(bst['directional_collision_vector'])==10
        assert bst['collision_proxy_kind']=='MeshLab_CGAL_Alpha_Wrap'
        assert bst['asset_closed_edge_topology'] and bst['collision_proxy_full_asset']
        assert not bst['collision_proxy_partial_asset']
        np.testing.assert_array_equal(bv,again[0])
        report.append(dict(task=task,original_noncontact_pairs=ast['collision_noncontact_pairs'],
            wrapped_noncontact_pairs=bst['collision_noncontact_pairs'],collision_cost=bst['collision_cost'],
            collision_known=bst['collision_cost_known'],continuous_resolved=bst['closed_asset_interior_proxy_mask'],
            support_identical=True,proxy_sha256=bst['collision_proxy_file_sha256']))
    out=clean(dict(status='passed',tasks=len(tasks),cold_seconds=first_seconds,warm_seconds=cached_seconds,
        no_labels_read=True,original_support_identical=True,pose_anchor_preserved=True,workers=processes,rows=report))
    (HERE/'wrapped_geometry_smoke.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
