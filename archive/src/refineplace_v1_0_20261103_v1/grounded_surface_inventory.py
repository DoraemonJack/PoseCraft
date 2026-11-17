import json
from pathlib import Path
from collections import Counter

OUT=Path(__file__).resolve().parent
LOCK=Path('/path/to/research300/official_split_training_audit_20261102_v1/official563_training_split_lock.json')
if not LOCK.exists():
    LOCK=Path('/path/to/research300/refineplace_v01_official563_inputs_20261102_v1/official563_training_split_lock.json')
def main():
    roots=[Path('/path/to/research100/strict_facing_real30_observations_20261102_v1/inputs')]
    roots += [p/'inputs' for p in sorted(Path('/path/to/research300/refineplace_v01_officialtrain_images_20261102_v1/batches').glob('batch*')) if (p/'frame_source_identity_before_models.json').is_file()]
    scenes={}
    for root in roots:
        for folder in root.glob('scene*'):
            frames=[str(p) for p in sorted((folder/'color').glob('*.jpg')) if (folder/'depth'/(p.stem+'.png')).exists() and (folder/'pose'/(p.stem+'.txt')).exists()]
            if folder.name not in scenes or len(scenes[folder.name]['frames'])<len(frames):
                scenes[folder.name]=dict(folder=str(folder),frames=frames)
    counts=Counter(); categories=Counter(); useful=Counter(); cache_scenes=set()
    for path in Path('/path/to/research300/refineplace_reference_clean_v2_batch_20261102_v1').glob('*.json'):
        if len(path.stem)!=24:continue
        row=json.loads(path.read_text())
        if not row.get('scene'):continue
        phrase=row.get('detector_phrase','').lower()
        counts[row.get('status')]+=1;categories[phrase]+=1
        if any(w in phrase for w in ('floor','table','desk','counter','cabinet','shelf','bed','chair','sofa','bench','stool')):
            useful[row.get('status')]+=1;cache_scenes.add(row['scene'])
    report=dict(frame_scenes=len(scenes),frames=sum(len(v['frames']) for v in scenes.values()),
        input_roots=[str(x) for x in roots], cache_status=dict(counts),bearing_cache_status=dict(useful),
        bearing_cache_scenes=len(cache_scenes),top_cache_phrases=categories.most_common(30),scenes=scenes)
    (OUT/'grounded_surface_inventory.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('scenes','input_roots')},indent=2))
if __name__=='__main__':main()
