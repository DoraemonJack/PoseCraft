import json, math
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent/'paired10k'
def nulls(v,path=''):
    if v is None: yield path
    elif isinstance(v,dict):
        for k,x in v.items():yield from nulls(x,path+'.'+k)
    elif isinstance(v,list):
        for x in v:yield from nulls(x,path+'[]')
out={}
for arm in ('baseline','upgraded'):
    records=[json.loads(x) for x in (ROOT/(arm+'_test300')/'predictions_in_progress.jsonl').read_text().splitlines() if x.strip()]
    paths=Counter(p for r in records for p in nulls(r))
    examples=[]
    for r in records:
        for p in r['path']:
            g=p.get('geometry',{});meta=p.get('meta',{})
            distances=g.get('support_distances_m',[])
            logits=meta.get('mode_logits',[])
            if None in distances or None in logits:
                examples.append(dict(entry=r['original_entry_id'],step=p['step'],mode=p['mode'],
                     null_support_distances=distances.count(None),support_samples=len(distances),
                     known=g.get('support_known_fraction'),unknown=g.get('support_unknown_fraction'),
                     mode_logits=logits,probabilities=meta.get('mode_policy_probs')))
    failures=[]
    for p in ROOT.glob(arm+'_test300_failed_logging_*'):
        rows=[json.loads(x) for x in (p/'predictions_in_progress.jsonl').read_text().splitlines() if x.strip()]
        failures.append(dict(directory=p.name,rows=len(rows),last_id=rows[-1]['original_entry_id'] if rows else None))
    out[arm]=dict(records=len(records),null_paths=dict(paths),examples=examples[:8],failed_logs=failures,
       final_pose_all_finite=all(math.isfinite(r['prediction'][k]) for r in records for k in ('tx','ty','tz','rotation')))
print(json.dumps(out,indent=2))
