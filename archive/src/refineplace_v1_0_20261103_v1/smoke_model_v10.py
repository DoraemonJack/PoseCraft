"""CPU integration check: real train observation, real previous weights/labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

import train_v10 as training
from model_v10 import model_from_checkpoint
from orientation_policy_v1 import DirectionalReferencePosePolicy
from support_regions import FEATURE_NAMES
from v1_scene import V10Scene


def main(a):
    torch.set_num_threads(1)
    torch.manual_seed(419)
    d=training.d
    lock=d.read(d.LOCK)
    library=d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json')
    source=V10Scene(a.scene,lock,library,driver=d,grounded_root=a.grounded_root)
    try:
        rows=source.inputs([0,1])
        features=torch.tensor(np.stack([r['features'] for r in rows]))
        text=torch.tensor(np.stack([r['text'] for r in rows]))
        assets=torch.stack([torch.load(d.b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/
            (r['asset_uuid']+'.pt'),map_location='cpu',weights_only=True).reshape(-1) for r in rows])
        regions=torch.tensor(np.stack([r['region_features'] for r in rows]))
        batch,candidates=features.shape[:2]
        mask=torch.ones((batch,candidates),dtype=torch.bool)
        # Explicit all-mask isolates the new region branch; not a reference quality test.
        context=dict(candidate_xyz=torch.tensor(np.stack([r['centers'] for r in rows])),
            candidate_mask=mask,reference_xyz=torch.zeros(batch,1,3),reference_extents=torch.zeros(batch,1,3),
            reference_mask=torch.zeros(batch,1,dtype=torch.bool),reference_logits=torch.zeros(batch,1,2),
            relation_types=torch.zeros(batch,1,dtype=torch.long),relation_mask=torch.zeros(batch,1,dtype=torch.bool))
        types=d.read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types']
        checkpoint=torch.load(a.parent,map_location='cpu',weights_only=False)
        model=model_from_checkpoint(d,types,len(FEATURE_NAMES),checkpoint).eval()
        previous=DirectionalReferencePosePolicy(model.inner).eval()
        previous.load_state_dict(checkpoint['joint_policy'],strict=True)
        model.reference_head.inferred_mask=torch.zeros(batch,1,dtype=torch.bool)
        with torch.no_grad(),previous.reference_batch(**context):
            original=previous.candidates_batch(features,text,assets,mask)
        with torch.no_grad(),model.reference_batch(**context),model.region_batch(regions):
            initial=model.candidates_batch(features,text,assets,mask)
        errors=[float((left-right).abs().max()) for left,right in zip(original,initial)]
        assert errors==[0.,0.,0.],errors
        labels=source.labels(rows);keys=[(r['scene'],r['group_id']) for r in rows]
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        gradients=[];losses=[]
        model.train()
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            with model.reference_batch(**context),model.region_batch(regions):
                scores,fused,values=model.candidates_batch(features,text,assets,mask)
                offsets=model.offsets_batch(fused,mask,radius=.15)
                losses_batch,_=d.supervised_safe(scores,offsets,labels,keys)
                loss=losses_batch.mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            grads={n:float(p.grad.abs().sum()) for n,p in model.named_parameters()
                   if n.startswith('region_') and p.grad is not None}
            gradients.append(grads);losses.append(float(loss.detach()))
            optimizer.step()
        assert gradients[0]['region_pose_score.weight']>0
        assert gradients[1]['region_encoder.0.weight']>0
        report=dict(passed=True,source='actual official train scene and candidate observations',scene=a.scene,
            tasks=keys,parent_checkpoint=str(a.parent),region_shape=list(regions.shape),
            original_feature_reconstruction_max_error=source.feature_reconstruction_max_error,
            zero_initialized_region_output_max_error=errors,supervised_losses=losses,
            region_gradients=gradients,reference_scope='explicit all-mask isolation',
            limitations='CPU forward/supervised gradient smoke; not RL, GPU throughput or validation accuracy')
        a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report))
    finally:
        source.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--scene',default='scene0190_00')
    p.add_argument('--parent',type=Path,default=training.PARENT)
    p.add_argument('--grounded-root',type=Path,default=Path(__file__).parent/'grounded_support/grounded_surfaces')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'model_smoke.json')
    main(p.parse_args())
