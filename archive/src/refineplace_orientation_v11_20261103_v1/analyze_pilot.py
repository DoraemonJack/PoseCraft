"""Read-only diagnostics of the completed paired pilot; no test labels read."""
import json, math
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def average(xs):return sum(xs)/len(xs) if xs else None
def hist(angles):return dict(Counter(str(round(math.degrees(x)%360,4)) for x in angles))
manifest=json.loads((ROOT/'manifest.json').read_text())
facing={(t['scene'],t['group_id']):t['facing_requested'] for t in manifest['tasks']}
test_catalog=Path('/path/to/research300/refineplace_v1_epoch2_fulltest_20261103_v1/references/reference_catalog_compact_v4_test.jsonl')
catalog={r['original_entry_id']:r for r in map(json.loads,test_catalog.read_text().splitlines()) if r['original_entry_id']<300}
test_facing={i for i,r in catalog.items() if any(q['type']=='facing' for q in r['queries'])}
report={}
for arm in ('baseline','upgraded'):
    folder=ROOT/'paired10k'/arm
    rows=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
    moves=Counter()
    for r in rows:moves.update(r['action_modes'])
    ang=[a for r in rows for a in r['angular_reward_diagnostics']]
    facing_ang=[a for r in rows for key,a in zip(r['task_ids'],r['angular_reward_diagnostics']) if facing[tuple(key)]]
    positive=[v for r in rows for v in r['yaw_diagnostics'].get('positive_count',[])]
    phase={}
    for name,part in [('first100',rows[:100]),('last100',rows[-100:])]:
        total=sum(r['tasks'] for r in part)
        phase[name]={k:sum(r[k]*r['tasks'] for r in part)/total for k in ('loss','supervised_loss','rl_loss','collision_aux_loss','task_quality')}
    jr=ROOT/'paired10k'/(arm+'_test300')/'predictions_in_progress.jsonl'
    records=[json.loads(x) for x in jr.read_text().splitlines()]
    probs=[p['meta']['mode_policy_probs'] for r in records for p in r['path']]
    train=dict(tasks=sum(r['tasks'] for r in rows),facing_tasks=sum(r['facing_tasks'] for r in rows),action_modes=dict(moves),
        angular_enabled=sum(a.get('enabled',False) for a in ang),
        facing_within_half_meter=sum(a.get('distance',float('inf'))<=.5 for a in facing_ang) if arm=='upgraded' else None,
        facing_quality_over_1e_minus4=sum(a.get('quality',0)>1e-4 for a in facing_ang),
        positive_candidate_tasks=sum(p>0 for p in positive) if positive else None,zero_positive_tasks=sum(p==0 for p in positive) if positive else None,
        first_collision_multiplier=rows[0]['collision_multiplier_used'],last_collision_multiplier=rows[-1]['collision_multiplier_used'],phase=phase)
    test=dict(tasks=len(records),scene_counts=dict(Counter(r['scene'] for r in records)),
        parsed_facing_tasks=len(test_facing),parsed_facing_final_yaw=hist([r['yaw'] for r in records if r['original_entry_id'] in test_facing]),
        asset_counts=dict(Counter(r['asset_uuid'] for r in records)),
        reference_active=sum(r['reference_active'] for r in records),
        initial_zero_yaw=sum(abs(r['initial_yaw'])<1e-6 for r in records),final_zero_yaw=sum(abs(r['yaw'])<1e-6 for r in records),
        changed_yaw=sum(abs(math.atan2(math.sin(r['yaw']-r['initial_yaw']),math.cos(r['yaw']-r['initial_yaw'])))>1e-6 for r in records),
        move_probability_mean=average([p[0] for p in probs]),move_probability_max=max(p[0] for p in probs),
        median_move_probability=sorted(p[0] for p in probs)[len(probs)//2],
        max_angle_change_deg=max(abs(math.degrees(math.atan2(math.sin(r['yaw']-r['initial_yaw']),math.cos(r['yaw']-r['initial_yaw'])))) for r in records),
        yaw_per_asset={uid:hist([r['yaw'] for r in records if r['asset_uuid']==uid]) for uid in sorted({r['asset_uuid'] for r in records})})
    report[arm]=dict(train=train,test=test)
(ROOT/'paired_diagnostics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
