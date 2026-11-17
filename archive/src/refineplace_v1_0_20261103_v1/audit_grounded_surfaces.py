"""Validate frozen image support cache completeness and provenance before admission."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from build_grounded_surfaces import sha,write,LABELS

def audit(root):
    out=root/'grounded_surfaces';index=json.loads((out/'index.json').read_text())
    detections=[json.loads(p.read_text()) for p in (root/'detections').glob('scene*.json')]
    status=Counter(r['status'] for r in detections)
    assert not status['failed'],status
    assert index['status']=='complete' and index['no_GT_labels']
    expected=index.get('expected_train_scenes',len(index.get('expected_validation_scenes',[])))
    assert index['scenes']==expected,(index['scenes'],expected)
    assert not index['missing_frame_scenes']
    labels=Counter();observations=Counter();total=0;matched=0;projected=0
    checks=[]
    for rec in sorted(index['records'],key=lambda x:x['scene']):
        scene=rec['scene'];path=out/(scene+'.npz');meta=json.loads((out/(scene+'.json')).read_text())
        assert sha(path)==meta['npz_sha256']
        assert meta['no_ground_truth_semantic_or_instance_labels']
        assert meta['coordinate_frame']=='original_scene_vertex_ids'
        assert meta['source']=='grounding_dino_sam2'
        with np.load(path,allow_pickle=False) as z:
            ids=z['point_ids'];lid=z['label_ids'];conf=z['confidence'];n=z['observation_counts']
            assert len(ids)==len(lid)==len(conf)==len(n)==meta['point_label_records']
            assert np.isfinite(conf).all() and np.all((conf>=0)&(conf<=1))
            assert np.all((ids>=0)&(ids<meta['mesh_vertices']))
            assert np.all((lid>=0)&(lid<len(LABELS)))
            assert np.all((n>=1)&(n<=max(meta['frames'],1)))
            assert len(np.unique(np.stack([ids,lid],axis=1),axis=0))==len(ids)
            labels.update({LABELS[k]:int((lid==k).sum()) for k in range(len(LABELS))})
            observations.update({str(k):int((n==k).sum()) for k in np.unique(n)})
            total+=len(ids)
        matched+=meta['matched_samples'];projected+=meta['projected_samples']
        checks.append({'scene':scene,'records':meta['point_label_records'],'frames':meta['frames']})
    result=dict(status='passed',partition=index.get('partition','official_train563'),scenes=index['scenes'],
        scenes_with_surface_evidence=index['scenes_with_surface_evidence'],
        scenes_without_surface_evidence=index['scenes_without_surface_evidence'],
        frames=len(detections),detection_status=dict(status),point_label_records=total,
        label_records=dict(labels),observation_counts=dict(observations),
        projected_samples=projected,matched_samples=matched,
        overall_projection_to_mesh_agreement=matched/projected if projected else None,
        data_index_sha256=sha(out/'index.json'),
        no_GT_semantic_instance_or_placement_labels=True,
        projection_caveat='Identity RGB/depth extrinsic approximation retained; 40mm nearest-mesh filtering with soft confidence, not calibration certification.',
        checks=checks)
    write(root/'admission_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();audit(a.root)
