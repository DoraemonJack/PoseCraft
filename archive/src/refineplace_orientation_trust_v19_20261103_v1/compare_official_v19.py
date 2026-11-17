import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
 paths=dict(V13=HERE.parent/'refineplace_orientation_v13_20261103_v1/refine1000_v1/official_test300/summary.json',
  V15=HERE.parent/'refineplace_orientation_v15_20261103_v1/full8/official_test300/summary.json',
  V19=HERE/'projected_run/official_test300/summary.json')
 data={k:json.loads(p.read_text()) for k,p in paths.items()}
 assert all(r['status']=='complete' and r['tasks']==300 and r['image']==data['V13']['image'] for r in data.values())
 results={}
 for k,r in data.items():
  m=r['official_metrics'];c=m['per_constraint_metrics'];results[k]=dict(complete_percent=m['complete_placement_success'],complete_count=round(m['complete_placement_success']*3),facing=c['facing'],physical=c['plausible'])
 report=dict(status='complete',tasks=300,results=results,delta_complete_percentage_points_vs_V13=results['V19']['complete_percent']-results['V13']['complete_percent'],
  delta_complete_percentage_points_vs_V15=results['V19']['complete_percent']-results['V15']['complete_percent'],
  scope='Same previously exposed official TEST IDs 0..299; diagnostic comparison. V19 warm-starts V13 semantic scoring with repair-protected gradients; original risk and position remain frozen.',
  hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})
 (HERE/'official_comparison.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(json.dumps(main()))
