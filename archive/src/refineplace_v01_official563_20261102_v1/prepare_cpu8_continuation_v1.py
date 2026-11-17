"""Prepare-only admission for an explicit CPU8 continuation. NEVER launches training.
The v2 resume protocol is deliberately not bypassed or edited here.
"""
import argparse, hashlib, json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 a=argparse.ArgumentParser();a.add_argument('--output-plan',required=True);args=a.parse_args()
 live=Path('/path/to/research300/refineplace_v01_official563_joint_full_20261102_v1')
 comparison=Path('/path/to/research300/refineplace_v01_training_step_pool_comparison_v2_20261102_v1/summary.json')
 cold=Path('/path/to/research300/refineplace_v01_cpu8_cold_resume_validation_20261102_v1/summary.json')
 workers=Path('/path/to/research100/refineplace_persistent_grouped_pool_modules_v2')
 expected='be17c8c46ed37cabda5ece88a591a10af7576dc7aaae518d910cacae46f8cf4c'
 assert read(live/'summary.json')['status']=='stopped_checkpoint_saved'
 assert (live/'stop_after_current_batch').exists() and sha(live/'last_step.pt')==expected
 assert sha(comparison)=='02484782db018c593d513b405489c88e889809cee786697efd20ee4c7ed6a696'
 comp=read(comparison);recovery=read(cold)
 assert comp['status']=='actual_multiscene_training_steps_parallel_exact'
 assert recovery['status']=='actual_cpu8_cold_cache_restart_exact'
 assert comp['formal_run_unchanged_and_still_stopped'] and recovery['formal_run_unchanged_and_still_stopped']
 assert recovery['arms']['process8']['all_geometry_forward_losses_gradients_parameters_optimizer_RNG_exact']
 assert read(cold.parent/'cold_restart_audit.json')['model_optimizer_all_RNG_restored_exactly']
 assert sha(workers/'persistent_grouped_geometry_v2.py')=='d10475ec23c497336e52cdfbb6bc4936789a80478821ea82f8d3a97de6417bb1'
 assert sha(workers/'private_worker_geometry_v1.py')=='4b214d2a0b2b6c41615cfd0aa0a0ecf773c81521a1997bac4d88240dc9b4eaee'
 protocol=live/'protocol_before_training.json'
 assert sha(protocol)=='3785c7b0f48c12ab0d053d24ec628298f8a284cc96a126f1ddf0d47557631d4f'
 plan=dict(status='prepared_only_NOT_launched',parent_checkpoint=str(live/'last_step.pt'),parent_checkpoint_sha256=expected,parent_protocol_sha256=sha(protocol),parent_next_cursor=[1,97,64],parent_updates=572,comparison_sha256=sha(comparison),cold_resume_summary_sha256=sha(cold),new_backend='persistent_grouped_CPU8_v2',workers=8,worker_source_sha256=sha(workers/'persistent_grouped_geometry_v2.py'),helper_sha256=sha(workers/'private_worker_geometry_v1.py'),unchanged_learning_protocol=read(protocol),required_new_driver_contract=['New version and new output directory; explicit parent checkpoint chain, never edit old protocol','Copy model, optimizer, all CPU/CUDA/numpy/python RNG and cursor exactly; record committed parent log prefix','ObservationScene does not construct unused parent BVH; one persistent private pool across all batches','Only query_many dispatch changes; restore original query order and propagate every exception','Keep source split/norm/catalog/task order, label boundary, all loss/reward/optimizer/frozen modules unchanged','close pool on stop/failure; checkpoint at completed batch; worker caches need no serialization'],training_launched=False,formal_run_modified=False,limitations='Admission/continuation plan only; the original v2 driver intentionally rejects a changed backend. A separately audited explicit-migration driver is still required before formal restart.')
 out=Path(args.output_plan);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(plan,ensure_ascii=False,indent=2));print(json.dumps(dict(status=plan['status'],output=str(out),sha256=sha(out))))
if __name__=='__main__':main()
