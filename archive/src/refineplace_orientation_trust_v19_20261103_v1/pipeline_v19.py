"""Infer only the dev-selected frozen checkpoint; no test-based model selection."""
import pathlib,subprocess,sys,json,hashlib
r=pathlib.Path(__file__).resolve().parent
out=r/'pipeline_v1';out.mkdir(exist_ok=False)
def write(x):
 p=out/'progress.tmp';p.write_text(json.dumps(x));p.replace(out/'progress.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 summary=json.loads((r/'projected_run/summary.json').read_text())
 assert summary['status']=='admitted' and summary['frozen_v13_unchanged'] and summary['risk_predictions_bitwise_unchanged']
 assert sha(r/'projected_run/final.pt')==summary['checkpoint_sha256']
 for n,folder in [(8,'smoke8'),(300,'test300')]:
  write(dict(stage='inference',tasks=n))
  cmd=[sys.executable,'-u',str(r/'infer_test300_v19.py'),'--checkpoint',str(r/'projected_run/final.pt'),'--training-protocol',str(r/'projected_run/protocol_before_training.json'),'--output',str(r/'projected_run'/folder),'--limit',str(n)]
  with (out/(folder+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 write(dict(stage='predictions_ready',status='complete',official_test300_complete=True))
except Exception as e:
 write(dict(stage='failed',status='failed',error=str(e)));raise
