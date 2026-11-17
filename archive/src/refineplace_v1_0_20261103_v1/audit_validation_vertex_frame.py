"""Read-only check that grounded IDs and old official coordinate export agree."""
import json
from pathlib import Path
import numpy as np
import torch
from build_grounded_surfaces import HERE,MESH,write

if __name__=='__main__':
    manifest=json.loads(MESH.read_text())
    grounded=HERE/'validation_grounded_support/grounded_surfaces'
    index=json.loads((grounded/'index.json').read_text())
    geom=json.loads(Path('/path/to/research100/native_val297_current_model_20261102_v1/allmask_current/geometry_provenance.json').read_text())
    root=Path('/path/to/home/codex_3d_research_20261028/rl_position_adjustment_v1/scans_full_refer_v1/val')
    rows=[]
    for rec in index['records']:
        scene=rec['scene'];payload=torch.load(root/(scene+'_refer.pth'),map_location='cpu',weights_only=False)
        xyz=np.asarray(payload[0],np.float32);del payload
        with np.load(manifest['scenes'][scene]['path'],allow_pickle=False) as z:world=np.asarray(z['vertices'],float)
        g=geom[scene];A=np.asarray(g['A'],float);translation=np.asarray(g['translation'],float)
        raw_error=float(np.linalg.norm(world-xyz-translation,axis=1).max())
        aligned=(xyz.astype(float)@A[:3,:3].T).astype(np.float32)
        offset=A[:3,:3]@translation+A[:3,3]
        export_error=float(np.linalg.norm(world@A[:3,:3].T+A[:3,3]-(aligned+offset),axis=1).max())
        meta=json.loads((grounded/(scene+'.json')).read_text())
        assert len(xyz)==meta['mesh_vertices'] and meta['mesh_sha256']==g['mesh_sha256']
        assert raw_error<1e-5 and export_error<2e-5
        rows.append(dict(scene=scene,vertices=len(xyz),raw_frame_max_error_m=raw_error,aligned_export_max_error_m=export_error))
    report=dict(status='passed',scenes=len(rows),no_model_or_labels_used=True,records=rows)
    write(HERE/'validation_vertex_frame_audit.json',report)
    print(json.dumps({'status':'passed','scenes':len(rows),'max_export_error_m':max(r['aligned_export_max_error_m'] for r in rows)}))
