import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
import train_official563_joint_v7_sync_full as d
from v1_scene import V10Scene,join_canonical_masks
torch.set_num_threads(1)
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--scenes',nargs='+',default=['scene0000_00','scene0001_00'])
p.add_argument('--grounded-root',default=str(root/'grounded_support_smoke/grounded_surfaces_cached'))
p.add_argument('--out',default=str(root/'support_scene_smoke.json'))
args=p.parse_args()
library=d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json')
lock=d.read(d.LOCK);reports=[]
check=join_canonical_masks([7,2,5],[2,2,7],np.eye(8,dtype=bool)[[0,3,1]])
assert check[0,1] and check[1,0] and check[1,3] and not check[2].any()
for scene in args.scenes:
    started=time.perf_counter()
    s=V10Scene(scene,lock,library,driver=d,grounded_root=args.grounded_root,cache_root=root/'support_smoke_cache')
    assert s.labels_arrays is None
    rows=s.inputs([0,1]);assert s.labels_arrays is None
    ids_before=[r['candidate_ids'].copy() for r in rows]
    labels=s.labels(rows)
    for r,ids in zip(rows,ids_before):
        assert np.array_equal(r['candidate_ids'],ids)
        target=labels[scene,r['group_id']]
        assert target['positive'].shape==(256,8) and target['rank'].shape==(256,8)
        assert np.isfinite(target['rank']).all() and target['offset_xy'].shape==(256,8,2)
        if target['positive'].any():
            assert np.max(np.abs(target['offset_xy'][target['positive']]))<1e-4
    report=dict(scene=scene,seconds=time.perf_counter()-started,
        source_feature_error=s.feature_reconstruction_max_error,
        geometry=s.prepared['report'],features=[list(r['features'].shape) for r in rows],
        region_features=[list(r['region_features'].shape) for r in rows],
        proposals=[r['region_report'] for r in rows],
        positive_pose_counts=[int(labels[scene,r['group_id']]['positive'].sum()) for r in rows],
        candidates_frozen_before_label_read=True)
    reports.append(report);print(json.dumps(report),flush=True);s.close()
Path(args.out).write_text(json.dumps(reports,indent=2))
