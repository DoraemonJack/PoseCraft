import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = Path('/path/to/research300/refineplace_sync20_inferred_official150_20261102_v1/fixed_validation150.json')
ROOTS = [
    '/path/to/research100/train8_image_reference_pilot_20261102_v1/inputs',
    '/path/to/research100/native_val50_current_model_20261102_v1/real_image_observations/inputs',
    '/path/to/research100/newdev_rgbd_ref_audit_20261102_v1/inputs',
    '/path/to/research100/strict_facing_real30_observations_20261102_v1/inputs',
    '/path/to/research100/strict_facing_real8_20261102_v1/inputs',
    '/path/to/research100/cross_category_real32_observations_20261102_v1/inputs',
    '/path/to/research300/partial_support_head_after_mining_20261101_v1/inputs',
    '/path/to/research300/horizontal_support_after_mining_20261101_v1/inputs',
]

if __name__ == '__main__':
    manifest=json.loads(MANIFEST.read_text())
    tasks=manifest['tasks']
    scenes=sorted({t['scene'] for t in tasks})
    records={}
    for root in ROOTS:
        for scene in scenes:
            folder=Path(root)/scene
            if not (folder/'intrinsics_color.txt').exists() or not (folder/'intrinsics_depth.txt').exists():
                continue
            frames=[str(p) for p in sorted((folder/'color').glob('*.jpg'))
                    if (folder/'depth'/(p.stem+'.png')).exists() and (folder/'pose'/(p.stem+'.txt')).exists()]
            if len(frames)>len(records.get(scene,{}).get('frames',[])):
                records[scene]={'folder':str(folder),'frames':frames}
    result={'manifest':str(MANIFEST),'tasks':len(tasks),'expected_scenes':scenes,'scenes':records,
            'missing_scenes':sorted(set(scenes)-set(records)),
            'frames':sum(len(r['frames']) for r in records.values())}
    (HERE/'validation_surface_inventory.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='scenes'},indent=2))
