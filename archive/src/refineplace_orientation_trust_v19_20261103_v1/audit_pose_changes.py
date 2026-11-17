"""Observation-only comparison; no per-task official labels are read."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATTERN = re.compile(r'\b(?:facing|angled towards?|oriented towards?|pointing towards?|turned towards?|directed at|looking at)\b', re.I)

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def changed(a, b):
    return abs((a-b+180)%360-180) > 1e-4

def main():
    tasks = {r['original_entry_id']:r for r in read(HERE/'tasks_manifest_observation.json')['tasks']}
    paths = {'V13':HERE.parent/'refineplace_orientation_v13_20261103_v1/refine1000_v1/test300/predictions_in_progress.jsonl',
             'V19':HERE/'projected_run/test300/predictions_in_progress.jsonl'}
    traces = {k:{r['original_entry_id']:r for r in map(json.loads,p.read_text().splitlines())} for k,p in paths.items()}
    assert all(set(v)==set(range(300)) for v in traces.values())
    rows=[]
    for i in range(300):
        a,b=traces['V13'][i],traces['V19'][i];t=tasks[i]
        assert a['prompt_sha256']==b['prompt_sha256']==t['prompt_sha256']
        assert hashlib.sha256(t['prompt'].encode()).hexdigest()==t['prompt_sha256']
        assert a['frozen_v11_prediction']==b['frozen_v11_prediction']
        assert all(a['prediction'][k]==b['prediction'][k] for k in ('tx','ty','tz'))
        row=dict(entry_id=i,prompt=t['prompt'],explicit_facing_language=bool(PATTERN.search(t['prompt'])),
                 baseline_yaw=a['frozen_v11_prediction']['rotation'],v13_yaw=a['prediction']['rotation'],v19_yaw=b['prediction']['rotation'])
        row['changed_v13_to_v19']=changed(row['v13_yaw'],row['v19_yaw'])
        for name,tr in [('V13',a),('V19',b)]:
            row[name+'_rotated_from_v11']=changed(row['baseline_yaw'],tr['prediction']['rotation'])
            row[name+'_unknown_risk_steps']=sum(s['meta']['uncertain_collision_evidence'] for s in tr['refinement_path'])
            row[name+'_steps']=len(tr['refinement_path'])
        rows.append(row)
    counts={}
    for group in ('all','explicit_facing','other_language'):
        part=[r for r in rows if group=='all' or r['explicit_facing_language']==(group=='explicit_facing')]
        counts[group]={'tasks':len(part),**{k:sum(r[k] for r in part) for k in ('changed_v13_to_v19','V13_rotated_from_v11','V19_rotated_from_v11','V13_unknown_risk_steps','V19_unknown_risk_steps','V13_steps','V19_steps')}}
    result=dict(counts=counts,positions_identical=True,rows=rows,
                language_grouping='Prompt regex only; absence of facing words does not mean yaw cannot affect other conditions.',
                official_per_task_outcomes_available=False,
                provenance={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})
    (HERE/'pose_change_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
